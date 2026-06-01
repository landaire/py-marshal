{
  description = "py27-marshal - a Rust port of CPython's marshal.c";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    rust-overlay.url = "github:oxalica/rust-overlay";
    flake-utils.url = "github:numtide/flake-utils";
    crane.url = "github:ipetkov/crane";
  };

  outputs = {
    self,
    nixpkgs,
    rust-overlay,
    flake-utils,
    crane,
    ...
  }:
    flake-utils.lib.eachDefaultSystem (system: let
      overlays = [(import rust-overlay)];
      pkgs = import nixpkgs {inherit system overlays;};

      rustToolchainToml = fromTOML (builtins.readFile ./rust-toolchain.toml);
      inherit (rustToolchainToml.toolchain) channel components;

      # minimal (rustc + cargo + rust-std) keeps CI lean; the default profile
      # pulls rust-docs on every fresh runner. The components list from
      # rust-toolchain.toml (rustfmt, clippy) is added as extensions.
      rustToolchain = pkgs.rust-bin.stable.${channel}.minimal.override {
        extensions = components;
      };

      craneLib = (crane.mkLib pkgs).overrideToolchain rustToolchain;

      commonArgs = {
        src = craneLib.cleanCargoSource ./.;
        strictDeps = true;
      };

      cargoArtifacts = craneLib.buildDepsOnly commonArgs;
    in
      with pkgs; {
        packages.default = craneLib.buildPackage (commonArgs // {inherit cargoArtifacts;});

        checks = {
          inherit (self.packages.${system}) default;

          clippy = craneLib.cargoClippy (commonArgs
            // {
              inherit cargoArtifacts;
              cargoClippyExtraArgs = "--all-targets --all-features -- -D warnings";
            });

          fmt = craneLib.cargoFmt commonArgs;

          test = craneLib.cargoTest (commonArgs // {inherit cargoArtifacts;});
        };

        devShells.default = craneLib.devShell {
          packages = [
            cargo-edit
          ];
        };
      });
}
