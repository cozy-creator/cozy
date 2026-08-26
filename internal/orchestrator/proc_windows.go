//go:build windows

// The process group's Windows analogue: a JOB OBJECT. Windows has no process group a
// signal can address, so the tree is held by an object instead — every process the worker
// spawns is inside the job, `TerminateJobObject` ends all of them at once, and
// `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` means the tree cannot outlive this service even if
// it dies without stopping anything.
//
// The one structural difference from Unix is WHEN: `setpgid` is a fork-time flag, and a job
// can only take a process that already exists. The gap is closed by CREATE_SUSPENDED —
// the child is created but its first instruction has not run, the job adopts it, and only
// then is it resumed. Containment therefore FAILS CLOSED (#449): a child the job cannot
// take is terminated before it executes anything, and the worker never runs uncontained.
//
// Shutdown has two tiers, like Unix (#449): TERM is a CTRL_BREAK console event to the
// child's own process group — the cooperative ask the supervisor handles the way it
// handles SIGTERM elsewhere — and KILL is TerminateJobObject. The job handle lives until
// the process is provably gone (killed or reaped), so a reused pid can never collide with
// a stale entry.
package orchestrator

import (
	"fmt"
	"os/exec"
	"sync"
	"syscall"
	"unsafe"

	"golang.org/x/sys/windows"
)

var jobs = struct {
	sync.Mutex
	byPID map[int]windows.Handle
}{byPID: map[int]windows.Handle{}}

// setProcessGroup gives the child its own console process group — the address CTRL_BREAK
// needs — and creates it SUSPENDED, so the job can adopt it before it runs.
func setProcessGroup(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{
		CreationFlags: windows.CREATE_NEW_PROCESS_GROUP | windows.CREATE_SUSPENDED,
	}
}

// adoptProcessGroup puts the SUSPENDED child in a fresh job object and resumes it. Any
// failure before the resume terminates the child and returns the error: the child has not
// executed an instruction yet, so ending it is a clean refusal, not a kill.
func adoptProcessGroup(cmd *exec.Cmd) error {
	if cmd.Process == nil {
		return fmt.Errorf("no process to adopt")
	}
	pid := cmd.Process.Pid
	failClosed := func(err error) error {
		_ = cmd.Process.Kill()
		return fmt.Errorf("job-object containment failed for pid %d: %w", pid, err)
	}
	job, err := windows.CreateJobObject(nil, nil)
	if err != nil {
		return failClosed(err)
	}
	info := windows.JOBOBJECT_EXTENDED_LIMIT_INFORMATION{
		BasicLimitInformation: windows.JOBOBJECT_BASIC_LIMIT_INFORMATION{
			LimitFlags: windows.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
		},
	}
	if _, err := windows.SetInformationJobObject(
		job,
		windows.JobObjectExtendedLimitInformation,
		uintptr(unsafe.Pointer(&info)),
		uint32(unsafe.Sizeof(info)),
	); err != nil {
		_ = windows.CloseHandle(job)
		return failClosed(err)
	}
	handle, err := windows.OpenProcess(
		windows.PROCESS_SET_QUOTA|windows.PROCESS_TERMINATE, false, uint32(pid))
	if err != nil {
		_ = windows.CloseHandle(job)
		return failClosed(err)
	}
	defer windows.CloseHandle(handle)
	if err := windows.AssignProcessToJobObject(job, handle); err != nil {
		_ = windows.CloseHandle(job)
		return failClosed(err)
	}
	if err := resumeProcess(pid); err != nil {
		_ = windows.TerminateJobObject(job, 1)
		_ = windows.CloseHandle(job)
		return fmt.Errorf("job-object containment failed for pid %d: %w", pid, err)
	}
	jobs.Lock()
	jobs.byPID[pid] = job
	jobs.Unlock()
	return nil
}

// resumeProcess resumes every thread of a CREATE_SUSPENDED child — exactly one exists,
// because the process has never run. os/exec closes the creation-time thread handle, so
// the thread is found again by snapshot.
func resumeProcess(pid int) error {
	snap, err := windows.CreateToolhelp32Snapshot(windows.TH32CS_SNAPTHREAD, 0)
	if err != nil {
		return err
	}
	defer windows.CloseHandle(snap)
	var entry windows.ThreadEntry32
	entry.Size = uint32(unsafe.Sizeof(entry))
	resumed := 0
	for err = windows.Thread32First(snap, &entry); err == nil; err = windows.Thread32Next(snap, &entry) {
		if entry.OwnerProcessID != uint32(pid) {
			continue
		}
		th, err := windows.OpenThread(windows.THREAD_SUSPEND_RESUME, false, entry.ThreadID)
		if err != nil {
			return err
		}
		if _, err := windows.ResumeThread(th); err != nil {
			_ = windows.CloseHandle(th)
			return err
		}
		_ = windows.CloseHandle(th)
		resumed++
	}
	if resumed == 0 {
		return fmt.Errorf("no thread of pid %d was found to resume", pid)
	}
	return nil
}

// killGroup is the two shutdown tiers. TERM asks: a CTRL_BREAK event to the child's own
// console group, which its supervisor handles as the cooperative stop — the job stays,
// because asking is not ending. KILL ends the whole job and retires its handle.
func killGroup(pid int, sig syscall.Signal) error {
	if sig != syscall.SIGKILL {
		// The cooperative tier. A detached service shares no console with the child, in
		// which case this errors and the caller's bounded wait falls through to KILL —
		// the same shape as a Unix worker that ignores its SIGTERM.
		return windows.GenerateConsoleCtrlEvent(windows.CTRL_BREAK_EVENT, uint32(pid))
	}
	jobs.Lock()
	job, ok := jobs.byPID[pid]
	if ok {
		delete(jobs.byPID, pid)
	}
	jobs.Unlock()
	if ok {
		err := windows.TerminateJobObject(job, 1)
		_ = windows.CloseHandle(job)
		return err
	}
	// No job: a reconciled orphan from a previous life of this service. Ending the one
	// process is all a pid can buy.
	handle, err := windows.OpenProcess(windows.PROCESS_TERMINATE, false, uint32(pid))
	if err != nil {
		return err
	}
	defer windows.CloseHandle(handle)
	return windows.TerminateProcess(handle, 1)
}

// releaseGroup retires the job of a process that is already gone. Closing the handle also
// takes any straggling grandchildren with it (KILL_ON_JOB_CLOSE) — the same "the tree dies
// with the worker" promise the job exists for — and frees the pid slot so a later process
// reusing the pid can never collide with a stale job.
func releaseGroup(pid int) {
	jobs.Lock()
	job, ok := jobs.byPID[pid]
	if ok {
		delete(jobs.byPID, pid)
	}
	jobs.Unlock()
	if ok {
		_ = windows.CloseHandle(job)
	}
}

// alive asks whether the pid still names a running process. A handle that opens and reports
// STILL_ACTIVE is the closest Windows has to Unix's signal 0.
func alive(pid int) bool {
	handle, err := windows.OpenProcess(windows.PROCESS_QUERY_LIMITED_INFORMATION, false, uint32(pid))
	if err != nil {
		return false
	}
	defer windows.CloseHandle(handle)
	var code uint32
	if err := windows.GetExitCodeProcess(handle, &code); err != nil {
		return false
	}
	return code == 259 // STILL_ACTIVE
}
