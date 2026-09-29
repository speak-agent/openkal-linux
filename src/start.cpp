// Program startup, for a program that carries no runtime of its own.
//
// Something must receive control from the kernel, find the arguments the kernel
// left on the stack, establish the thread pointer, and call the program. Where
// a program already carries a runtime, that runtime's first object does it and
// this file is not reached: the definition below is used only when the linker
// has an undefined `_start' to satisfy, which happens exactly when no other
// object provides one.
//
// It belongs to the implementation rather than to whatever sits above, and the
// reason is visible in what it does. Every step is a fact about this kernel ---
// the layout of the stack at inception, the program headers the kernel reports,
// the instruction that sets the thread pointer. A consumer that contained these
// steps would contain a copy of them per environment, which is what depending
// on openkal was meant to remove.
//
// Control is handed on through `__libc_start_main', which is the name the
// arrangement already has a name for. The symbol is weak: a program written
// directly against openkal has none, and then this file runs the initialisers
// and calls `main' itself.
#ifdef OKL_STANDALONE

#include "sys.h"
#include "tls.h"
#include <openkal/memory.h>
#include <openkal/abort.h>

namespace okl {
void record(int argc, char** argv, char** envp);
okl_ulong auxval(okl_ulong key);
}

extern "C" {

int main(int, char**, char**);

[[gnu::weak]] int __libc_start_main(int (*)(int, char**, char**), int, char**,
                                    void (*)(), void (*)(), void (*)());

[[noreturn]] void __okl_start_c(long* sp);

}

namespace {

// The initialiser arrays the linker builds. Each is weak: a program with no
// initialisers has neither symbol, and the loop then runs zero times rather
// than failing to link.
using initialiser = void (*)(int, char**, char**);
extern "C" {
[[gnu::weak]] extern initialiser __preinit_array_start[];
[[gnu::weak]] extern initialiser __preinit_array_end[];
[[gnu::weak]] extern initialiser __init_array_start[];
[[gnu::weak]] extern initialiser __init_array_end[];
}

void run_initialisers(int argc, char** argv, char** envp) {
    for (initialiser* p = __preinit_array_start; p != __preinit_array_end; ++p) (*p)(argc, argv, envp);
    for (initialiser* p = __init_array_start;    p != __init_array_end;    ++p) (*p)(argc, argv, envp);
}

void* allocate(okl_uptr n, okl_uptr a) { return kal_alloc(n, a); }

// A PROGRAM LINKED POSITION-INDEPENDENT AND STATIC RELOCATES ITSELF, HERE.
//
// `-static-pie' is how a program with no interpreter still states the names it
// offers --- its dynamic symbol table, which is what a loader resolves a loaded
// object's references against (openkal 0.15, SPEC clause 11 entry 21). The
// kernel places such a program at an address of its choosing, and the words
// that hold an address inside the program --- a table of functions, a
// pointer in initialised data --- hold the address it was linked at until
// someone adds the difference. With no interpreter that is the program itself,
// before anything reads one of those words: so this runs first, reaches only
// what needs no relocation (its own locals, and two symbols the linker defines,
// hidden so that they are reached relative to the instruction), and calls
// nothing.
//
// A program linked at a fixed address has no dynamic section, `_DYNAMIC' is
// null, and nothing happens. The only relocations a static position-independent
// program carries are relative ones --- every name is bound when it is linked
// --- in the two forms linkers write: a table of entries, and the packed form
// (`-z pack-relative-relocs'). Anything else would mean the program was not
// what this code expects, and it stops rather than run on half-relocated.
struct elf_dyn  { long tag; okl_uptr val; };
struct elf_rela { okl_uptr offset; okl_uptr info; long addend; };
struct elf_ehdr { unsigned char ident[16]; unsigned short type, machine; unsigned version;
                  okl_uptr entry, phoff, shoff; unsigned flags;
                  unsigned short ehsize, phentsize, phnum, shentsize, shnum, shstrndx; };
struct elf_phdr { okl_u32 type, flags; okl_u64 offset, vaddr, paddr, filesz, memsz, align; };

extern "C" {
[[gnu::weak, gnu::visibility("hidden")]] extern const elf_dyn _DYNAMIC[];
[[gnu::weak, gnu::visibility("hidden")]] extern const elf_ehdr __ehdr_start;
}

#if defined(__x86_64__)
constexpr okl_uptr kRelative = 8;      // R_X86_64_RELATIVE
#elif defined(__aarch64__)
constexpr okl_uptr kRelative = 1027;   // R_AARCH64_RELATIVE
#elif defined(__riscv)
constexpr okl_uptr kRelative = 3;      // R_RISCV_RELATIVE
#endif

// How far from its link address the program was placed: where its header is,
// less the address its first segment states. Zero for a fixed address.
[[gnu::always_inline]] inline okl_uptr image_bias() {
    const elf_ehdr* eh = &__ehdr_start;
    if (eh == nullptr) return 0;
    const auto* ph = reinterpret_cast<const elf_phdr*>(reinterpret_cast<const unsigned char*>(eh) + eh->phoff);
    for (unsigned i = 0; i < eh->phnum; ++i)
        if (ph[i].type == 1 /* PT_LOAD */) return reinterpret_cast<okl_uptr>(eh) - static_cast<okl_uptr>(ph[i].vaddr);
    return 0;
}

[[gnu::always_inline]] inline void relocate_self(okl_uptr bias) {
    const elf_dyn* d = _DYNAMIC;
    if (d == nullptr) return;
    okl_uptr rela = 0, relasz = 0, relr = 0, relrsz = 0;
    for (; d->tag != 0; ++d) {
        if (d->tag == 7) rela = d->val;          // DT_RELA
        else if (d->tag == 8) relasz = d->val;   // DT_RELASZ
        else if (d->tag == 36) relr = d->val;    // DT_RELR
        else if (d->tag == 35) relrsz = d->val;  // DT_RELRSZ
    }
    const auto* r = reinterpret_cast<const elf_rela*>(bias + rela);
    for (okl_uptr i = 0; rela != 0 && i < relasz / sizeof(elf_rela); ++i) {
        if ((r[i].info & 0xffffffffu) != kRelative) okl::sys(okl::nr_exit_group, 127);
        *reinterpret_cast<okl_uptr*>(bias + r[i].offset) = bias + static_cast<okl_uptr>(r[i].addend);
    }
    // The packed form: an even entry is an address, relocated, and the words
    // after it are described by the odd entries that follow, one bit each.
    const auto* e = reinterpret_cast<const okl_uptr*>(bias + relr);
    okl_uptr* where = nullptr;
    for (okl_uptr i = 0; relr != 0 && i < relrsz / sizeof(okl_uptr); ++i) {
        if ((e[i] & 1) == 0) {
            where = reinterpret_cast<okl_uptr*>(bias + e[i]);
            *where++ += bias;
        } else {
            okl_uptr bits = e[i] >> 1;
            for (okl_uptr k = 0; bits != 0; ++k, bits >>= 1)
                if (bits & 1) where[k] += bias;
            where += 8 * sizeof(okl_uptr) - 1;
        }
    }
}

}  // namespace

#if defined(__x86_64__)
__asm__(
".text\n"
".globl _start\n"
".type _start,@function\n"
"_start:\n"
"  xor %ebp,%ebp\n"        // the outermost frame has no caller
"  mov %rsp,%rdi\n"        // the kernel left everything here
"  and $-16,%rsp\n"
"  call __okl_start_c\n"
"  hlt\n"
".size _start,.-_start\n"
);
#elif defined(__aarch64__)
__asm__(
".text\n"
".globl _start\n"
".type _start,%function\n"
"_start:\n"
"  mov x29,#0\n"
"  mov x30,#0\n"
"  mov x0,sp\n"
"  and x1,x0,#-16\n"
"  mov sp,x1\n"
"  bl __okl_start_c\n"
"  brk #0\n"
".size _start,.-_start\n"
);
#endif

extern "C" [[noreturn]] void __okl_start_c(long* sp) {
    const okl_uptr bias = image_bias();
    relocate_self(bias);

    const int argc = static_cast<int>(sp[0]);
    char** argv = reinterpret_cast<char**>(sp + 1);
    char** envp = argv + argc + 1;
    okl::record(argc, argv, envp);

    // The thread pointer must exist before anything that has thread-local
    // state runs, which includes the C library's own initialisation.
    okl::describe_tls(okl::auxval(3 /* AT_PHDR */),
                      okl::auxval(4 /* AT_PHENT */),
                      okl::auxval(5 /* AT_PHNUM */), bias);
    const okl::tls_block b = okl::make_tls(allocate);
    if (b.tp != nullptr) okl::set_thread_pointer(b.tp);

    if (__libc_start_main != nullptr) {
        __libc_start_main(main, argc, argv, nullptr, nullptr, nullptr);
        // A C library's hand-over does not return. Reaching here means one
        // did, and continuing would run the program a second time.
        kal_exit(127);
    }

    run_initialisers(argc, argv, envp);
    kal_exit(main(argc, argv, envp));
}

#endif  // OKL_STANDALONE
