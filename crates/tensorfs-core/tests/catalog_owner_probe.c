#define _GNU_SOURCE
#include <dlfcn.h>
#include <limits.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

/* A root broker must create SQLite sidecars under the Store owner's filesystem
 * identity. Reject the intermediate root-owned inode before SQLite's later
 * fchown can hide it from a normal metadata assertion. No SQLite fd is opened
 * or closed here, and no ownership or lock is changed by the probe. */
int fchown(int fd, uid_t uid, gid_t gid) {
    int (*original)(int, uid_t, gid_t) = dlsym(RTLD_NEXT, "fchown");
    char proc[64], path[PATH_MAX];
    snprintf(proc, sizeof(proc), "/proc/self/fd/%d", fd);
    ssize_t n = readlink(proc, path, sizeof(path) - 1);
    if (n > 0) {
        path[n] = 0;
        struct stat info;
        if (geteuid() == 0 && uid != 0 &&
            (strstr(path, "tensorfs.sqlite-wal") || strstr(path, "tensorfs.sqlite-shm")) &&
            fstat(fd, &info) == 0 && info.st_uid == 0) {
            fprintf(stderr, "root-owned SQLite sidecar before owner assignment: %s\n", path);
            _exit(91);
        }
    }
    return original(fd, uid, gid);
}
