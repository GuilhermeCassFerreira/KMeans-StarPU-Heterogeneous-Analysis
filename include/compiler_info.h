#ifndef COMPILER_INFO_H
#define COMPILER_INFO_H

// Compilador de host que gerou o binário. As macros são avaliadas na unidade que
// inclui este header: incluir no main de cada versão (compilado junto dos kernels de CPU).
#define KMEANS_STR2(x) #x
#define KMEANS_STR(x) KMEANS_STR2(x)

#if defined(__NVCOMPILER)
#define KMEANS_COMPILER "nvc++ " KMEANS_STR(__NVCOMPILER_MAJOR__) "." KMEANS_STR(__NVCOMPILER_MINOR__)
#elif defined(__clang__)
#define KMEANS_COMPILER "clang++ " __clang_version__
#elif defined(__GNUC__)
#define KMEANS_COMPILER "g++ " __VERSION__
#else
#define KMEANS_COMPILER "desconhecido"
#endif

// Marca no binário para conferir sem executar (usada pelos scripts de experimento):
//   grep -a -o -m1 'KMEANS_COMPILER=[[:print:]]*' <binário>
// O nvc++ descarta a variável se ninguém a usa (mesmo com "used"), por isso o
// log imprime o compilador a partir dela, via compiler_name().
__attribute__((used)) static const char kmeans_compiler_tag[] = "KMEANS_COMPILER=" KMEANS_COMPILER;

static inline const char *compiler_name() {
    return kmeans_compiler_tag + sizeof("KMEANS_COMPILER=") - 1;
}

#endif
