//go:build windows

// The process group's Windows analogue: a JOB OBJECT. Windows has no process group a
// signal can address, so the tree is held by an object instead — every process the worker
// spawns is inside the job, `TerminateJobObject` ends all of them at once, and
// `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` means the tree cannot outlive this service even if
// it dies without stopping anything.
//
// The one structural difference from Unix is WHEN: `setpgid` is a fork-time flag, and a job
// can only take a process that already exists. So the pair is split — `setProcessGroup`
// asks for a new console group before the start, and `adoptProcessGroup` puts the started
// process in its job.
package coord

import (
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

// setProcessGroup gives the child its own console process group, so a console event aimed
// at this service does not travel to it as a side effect.
func setProcessGroup(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{CreationFlags: windows.CREATE_NEW_PROCESS_GROUP}
}

// adoptProcessGroup puts the RUNNING child in a job object and remembers it by pid. A
// failure here is not fatal and is not silent either: the worker still runs, and
// `killGroup` falls back to ending the process it was given, which is what a caller with
// no job would have had anyway.
func adoptProcessGroup(cmd *exec.Cmd) {
	if cmd.Process == nil {
		return
	}
	job, err := windows.CreateJobObject(nil, nil)
	if err != nil {
		return
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
		return
	}
	handle, err := windows.OpenProcess(
		windows.PROCESS_SET_QUOTA|windows.PROCESS_TERMINATE, false, uint32(cmd.Process.Pid))
	if err != nil {
		_ = windows.CloseHandle(job)
		return
	}
	defer windows.CloseHandle(handle)
	if err := windows.AssignProcessToJobObject(job, handle); err != nil {
		_ = windows.CloseHandle(job)
		return
	}
	jobs.Lock()
	jobs.byPID[cmd.Process.Pid] = job
	jobs.Unlock()
}

// killGroup ends the whole job — every process the worker spawned, at once. `sig` carries
// no meaning here beyond "end it": Windows has no graceful group signal, so a TERM and a
// KILL are the same act and the CALLER's grace period is what separates them.
func killGroup(pid int, sig syscall.Signal) error {
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
	// No job: the process was adopted by nobody (a reconciled orphan from a previous life
	// of this service). Ending the one process is all a pid can buy.
	handle, err := windows.OpenProcess(windows.PROCESS_TERMINATE, false, uint32(pid))
	if err != nil {
		return err
	}
	defer windows.CloseHandle(handle)
	return windows.TerminateProcess(handle, 1)
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
