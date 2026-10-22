//! The calls a Memory makes to one service (custody, or the RAM tier), in the order it made
//! them, for that service's one sender thread: a stalled service delays only its own calls. A
//! call not yet sent can be cancelled (an Export by its page's Withdraw: its handle goes with
//! it, so the queue holds nothing beyond the pages it covers). Each call is a ticket; a waiter
//! can settle on the calls posted so far. Its lock is a leaf.

use std::collections::VecDeque;

use parking_lot::{Condvar, Mutex};

pub(crate) struct Outbox<C> {
    queue: Mutex<Queue<C>>,
    moved: Condvar,
}

struct Queue<C> {
    /// (ticket, bytes it frees, call), in order.
    calls: VecDeque<(u64, u64, C)>,
    next: u64,
    /// The ticket the sender took and has not finished.
    sending: Option<u64>,
    /// Bytes the calls posted and not yet sent free once sent (withdrawn kept pages).
    pending: u64,
    closed: bool,
}

impl<C> Default for Outbox<C> {
    fn default() -> Self {
        Outbox { queue: Mutex::new(Queue { calls: VecDeque::new(), next: 0, sending: None, pending: 0, closed: false }), moved: Condvar::new() }
    }
}

impl<C> Outbox<C> {
    /// Queues `call`; `bytes` are freed once it is sent. Returns its ticket.
    pub fn post(&self, call: C, bytes: u64) -> u64 {
        let mut q = self.queue.lock();
        let ticket = q.next;
        q.next += 1;
        q.pending += bytes;
        q.calls.push_back((ticket, bytes, call));
        drop(q);
        self.moved.notify_all();
        ticket
    }

    /// Removes the first queued call `matches` picks, unsent; true when there was one.
    pub fn cancel(&self, matches: impl Fn(&C) -> bool) -> bool {
        let mut q = self.queue.lock();
        let Some(i) = q.calls.iter().position(|(_, _, c)| matches(c)) else { return false };
        if let Some((_, bytes, _)) = q.calls.remove(i) {
            q.pending -= bytes;
        }
        drop(q);
        self.moved.notify_all();
        true
    }

    /// The next call to send (the sender's), `None` once closed and empty.
    pub fn next(&self) -> Option<(u64, u64, C)> {
        let mut q = self.queue.lock();
        loop {
            if let Some(call) = q.calls.pop_front() {
                q.sending = Some(call.0);
                return Some(call);
            }
            if q.closed {
                return None;
            }
            self.moved.wait(&mut q);
        }
    }

    /// The operation physically returned its bytes, before publishing its completion. The
    /// sender still owns the ticket and calls `sent(ticket, 0)` after releasing its owner.
    pub fn returned(&self, bytes: u64) {
        self.queue.lock().pending -= bytes;
    }

    /// The sender sent `ticket` (or gave up on it): its bytes are free.
    pub fn sent(&self, ticket: u64, bytes: u64) {
        let mut q = self.queue.lock();
        q.pending -= bytes;
        if q.sending == Some(ticket) {
            q.sending = None;
        }
        drop(q);
        self.moved.notify_all();
    }

    /// Blocks until every call posted before it has been sent (or cancelled).
    pub fn settle(&self) {
        let until = self.queue.lock().next;
        self.settle_to(until);
    }

    /// Blocks until every call with a ticket below `until` has been sent (or cancelled).
    pub fn settle_to(&self, until: u64) {
        let mut q = self.queue.lock();
        while q.sending.is_some_and(|t| t < until) || q.calls.front().is_some_and(|(t, ..)| *t < until) {
            self.moved.wait(&mut q);
        }
    }

    /// Bytes waiting on calls not yet sent.
    pub fn pending(&self) -> u64 {
        self.queue.lock().pending
    }

    pub fn try_pending(&self) -> Option<u64> { Some(self.queue.try_lock()?.pending) }

    /// Takes every call not yet sent (their sender is gone).
    pub fn drain(&self) -> Vec<C> {
        let mut q = self.queue.lock();
        q.pending = 0;
        q.calls.drain(..).map(|(.., c)| c).collect()
    }

    /// No more calls: the sender sends what is queued, then ends.
    pub fn close(&self) {
        self.queue.lock().closed = true;
        self.moved.notify_all();
    }
}
