#!/bin/bash
set -u
need="libnccl.so.2 libibverbs.so.1 libcublas.so.13 libcublasLt.so.13"
pins="8062c27c479c249292b48c4bc73af1deb22edee4dfff191fb48a7491ac506410  libcublas.so.13.6.0.2
4198b8726952fbf25a4e78f07178d66509aca4919b7324e935fe3915bff3efc8  libcublasLt.so.13.6.0.2"
known=" libc.so.6 libm.so.6 libpthread.so.0 libdl.so.2 libcuda.so.1 libcublas.so libcublasLt.so $need "
fail=0
names=$(grep -aohE 'lib[A-Za-z0-9_]+\.so(\.[0-9]+)*' /opt/tensorfold/bin/* | sort -u | tr '\n' ' ')
echo "libproof: engine dlopen names: $names"
for n in $names; do case "$known" in *" $n "*) ;; *) echo "libproof: UNCOVERED dlopen name $n"; fail=1 ;; esac; done
for s in $need; do
    f=$(ldconfig -p | awk -v s="$s" '$1 == s {print $NF; exit}')
    [ -n "$f" ] || { echo "libproof: $s MISSING"; fail=1; continue; }
    r=$(readlink -f "$f"); h=$(sha256sum "$r" | cut -c1-64); u=$(ldd "$r" 2>&1 | grep -c 'not found'); note=""
    [ "$u" = 0 ] || { note=" NEEDED NOT RESOLVED"; fail=1; }
    case $s in libcublas*) if printf '%s\n' "$pins" | grep -qx "$h  ${r##*/}"; then note="$note = pin"; else note="$note != PIN"; fail=1; fi ;; esac
    echo "libproof: $s -> $r ${h:0:12}$note"
done
ls /usr/lib/*-linux-gnu/libibverbs/ 2>/dev/null | grep -q mlx5 || { echo "libproof: mlx5 provider MISSING"; fail=1; }
[ "$fail" = 0 ] && echo "LIBPROOF PASS" || echo "LIBPROOF FAIL"
exit "$fail"
