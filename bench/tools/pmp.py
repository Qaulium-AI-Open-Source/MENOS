# Poor man's profiler (no root needed): gdb runs the target as its own child;
# an external loop sends SIGUSR2 to the target at a fixed interval; each stop
# records the stack and continues. Output: one ';'-joined stack per line.
import gdb, os
OUT = os.environ["PMP_OUT"]
gdb.execute("set pagination off")
gdb.execute("set print frame-arguments none")
gdb.execute("handle SIGUSR2 stop print nopass")
stacks = []
def record():
    frames = []
    f = gdb.newest_frame()
    first = True
    while f is not None:
        name = f.name() or "??"
        if "menos" in name and first:
            sal = f.find_sal()
            if sal.symtab is not None:
                name += "@" + sal.symtab.filename.split("/")[-1] + ":" + str(sal.line)
            first = False
        frames.append(name)
        f = f.older()
    stacks.append(";".join(reversed(frames)))
with open(os.environ["PMP_PIDFILE"], "w") as p:
    pass
gdb.execute("run")
while True:
    try:
        inf = gdb.selected_inferior()
        if inf.pid == 0:
            break
        record()
        gdb.execute("continue")
    except gdb.error:
        break
with open(OUT, "w") as out:
    out.write("\n".join(stacks) + "\n")
