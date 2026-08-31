# Nix development shell for GRCh38 variant-annotation tooling.
#
# Annotation strategy (honest, after three environment probes):
# - nixpkgs `vep` FAILS to build (transitive UCSC `kent` compile error).
# - SnpEff 4.3t (the only nixpkgs version) predates current GRCh38 gene
#   models, and the modern 5.x download endpoint is dead.
# - WORKING: the official Ensembl VEP container via podman (no kent issue,
#   current VEP, Ensembl's recommended deployment). See vep_data/vep_run.sh
#   + vep_data/vep_cache_download.sh.
#
# This shell provides the surrounding bioinformatics tools (VCF manipulation,
# indexing, compression) that the annotation pipeline needs.
{
  pkgs ? import <nixpkgs> { },
}:

pkgs.mkShell {
  buildInputs = with pkgs; [
    bcftools # VCF subset/sort/compress/annotate
    htslib # tabix/bgzip for indexed VCF handling
    samtools
  ];

  shellHook = ''
    cat <<'EOF'

    GRCh38 annotation tooling ready (bcftools/tabix/samtools).

    VEP annotation runs via the official Ensembl container (podman):
      bash vep_data/vep_cache_download.sh   # one-time ~25 GB GRCh38 cache
      bash vep_data/vep_run.sh              # annotate the challenge VCF

    EOF
  '';
}
