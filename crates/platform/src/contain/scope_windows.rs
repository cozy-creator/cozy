//! Windows: an executor's scope is a Job Object, which ends every member when it is ended or when
//! its last handle closes, the machine's death included. No scope outlives its machine, so a new
//! machine has none of an earlier one's to sweep.
#![allow(unsafe_code)]
use crate::{Error, Result};
use std::{
    io, mem,
    os::windows::io::{AsRawHandle, FromRawHandle, OwnedHandle},
};
use windows_sys::Win32::System::{
    JobObjects::{
        AssignProcessToJobObject, JOBOBJECT_BASIC_ACCOUNTING_INFORMATION,
        JOBOBJECT_BASIC_PROCESS_ID_LIST, JobObjectBasicAccountingInformation,
        JobObjectBasicProcessIdList, QueryInformationJobObject, TerminateJobObject,
    },
    Threading::{
        INFINITE, OpenProcess, PROCESS_SET_QUOTA, PROCESS_SYNCHRONIZE, PROCESS_TERMINATE,
        WaitForSingleObject,
    },
};

#[derive(Debug)]
pub struct Scope(OwnedHandle);

impl Scope {
    pub fn create(_: &str) -> Result<Self> {
        Ok(Self(crate::process::job().map_err(Error::os("create a process scope"))?))
    }

    pub fn sweep(_: &str) -> Result<usize> {
        Ok(0)
    }

    pub fn cgroup_relative(&self) -> Option<&str> {
        None
    }

    pub fn environment(&self) -> Option<(&'static str, &str)> {
        None
    }

    pub fn trampoline_args(&self) -> Vec<String> {
        vec!["--scope-backend".into(), "inherit".into()]
    }

    /// Live processes in the scope.
    pub fn processes(&self) -> Result<usize> {
        // SAFETY: plain-old-data the call fills.
        let mut accounting: JOBOBJECT_BASIC_ACCOUNTING_INFORMATION = unsafe { mem::zeroed() };
        self.query(JobObjectBasicAccountingInformation, &mut accounting)
            .map_err(Error::os("count a scope's processes"))?;
        Ok(accounting.ActiveProcesses as usize)
    }

    /// Moves a running process into the scope; its later children start inside it.
    pub fn adopt(&self, pid: u32) -> Result<()> {
        // SAFETY: plain call; a null handle is the failure.
        let process = unsafe { OpenProcess(PROCESS_SET_QUOTA | PROCESS_TERMINATE, 0, pid) };
        if process.is_null() {
            return Err(Error::os("adopt a process")(io::Error::last_os_error()));
        }
        // SAFETY: a new handle that is now ours.
        let process = unsafe { OwnedHandle::from_raw_handle(process) };
        // SAFETY: a live job and a live process handle with the rights it needs.
        if unsafe { AssignProcessToJobObject(self.0.as_raw_handle(), process.as_raw_handle()) } == 0 {
            return Err(Error::os("adopt a process")(io::Error::last_os_error()));
        }
        Ok(())
    }

    pub fn kill(&self) -> Result<()> {
        // SAFETY: a live job handle.
        if unsafe { TerminateJobObject(self.0.as_raw_handle(), 1) } == 0 {
            return Err(Error::os("end a process scope")(io::Error::last_os_error()));
        }
        Ok(())
    }

    /// Kill whatever outlived the executor and wait for each to exit. Returns how many processes
    /// were still inside.
    pub fn end(&self) -> Result<usize> {
        let stragglers = self.processes()?;
        if stragglers == 0 {
            return Ok(0);
        }
        // Opened before the kill, so a member's exit is waited on, never a later process's id.
        let members: Vec<OwnedHandle> = self.members()?.into_iter().filter_map(open_waitable).collect();
        self.kill()?;
        for member in members {
            // SAFETY: a live process handle with SYNCHRONIZE.
            unsafe { WaitForSingleObject(member.as_raw_handle(), INFINITE) };
        }
        Ok(stragglers)
    }

    fn members(&self) -> Result<Vec<u32>> {
        // A list header and room for this many ids; a scope is one executor's tree.
        #[repr(C)]
        struct List {
            head: JOBOBJECT_BASIC_PROCESS_ID_LIST,
            more: [usize; 255],
        }
        // SAFETY: plain-old-data the call fills.
        let mut list: List = unsafe { mem::zeroed() };
        self.query(JobObjectBasicProcessIdList, &mut list)
            .map_err(Error::os("list a scope's processes"))?;
        let count = list.head.NumberOfProcessIdsInList as usize;
        // SAFETY: the first id is the header's own; the rest follow it in `more`, as the call wrote them.
        let ids = unsafe { std::slice::from_raw_parts(list.head.ProcessIdList.as_ptr(), count.min(256)) };
        Ok(ids.iter().map(|id| *id as u32).collect())
    }

    fn query<T>(&self, class: i32, into: &mut T) -> io::Result<()> {
        // SAFETY: a live job and a writable record of the class's layout and size.
        let queried = unsafe {
            QueryInformationJobObject(
                self.0.as_raw_handle(),
                class,
                (into as *mut T).cast(),
                mem::size_of::<T>() as u32,
                std::ptr::null_mut(),
            )
        };
        if queried == 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }
}

fn open_waitable(pid: u32) -> Option<OwnedHandle> {
    // SAFETY: plain call; a null handle is the failure.
    let process = unsafe { OpenProcess(PROCESS_SYNCHRONIZE, 0, pid) };
    // SAFETY: a new handle that is now ours.
    (!process.is_null()).then(|| unsafe { OwnedHandle::from_raw_handle(process) })
}
