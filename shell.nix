let overlay1 = self: super:
{
  sundials1 = self.callPackage ./SparseSundials { };
};

myHaskellPackageOverlay = self: super: {
  myHaskellPackages = super.haskellPackages.override {
    overrides = hself: hsuper: rec {

   tasty-golden =
        let newTastyGoldenSrc = builtins.fetchTarball { url = "https://hackage.haskell.org/package/tasty-golden-2.3.3/tasty-golden-2.3.3.tar.gz";
          sha256 = "0wgcs4pqr30bp801cyrg6g551i7q0vjjmd9gmk5jy44fgdhb7kkl";
          };
            tg = hself.callCabal2nix "tasty-golden" newTastyGoldenSrc {};
          in
            super.haskell.lib.dontCheck tg;

    hmatrix-sundials1 = super.haskell.lib.dontCheck (
        hself.callCabal2nix "hmatrix-sundials" (builtins.fetchGit {
    url = "file:///Users/dom/fu-hmatrix-sundials";
           rev = "0ce1d42d3ce186f92fde1d3eb8f9cbb66856b2f9";
}) { sundials_arkode          = self.sundials1;
     sundials_cvode           = self.sundials1;
     klu                      = self.suitesparse;
     suitesparseconfig        = self.suitesparse;
     sundials_sunlinsolklu    = self.sundials1;
     sundials_sunmatrixsparse = self.sundials1;
      });

  Naperian = hself.callCabal2nix "Naperian" (builtins.fetchGit {
    url = "https://github.com/idontgetoutmuch/Naperian.git";
    rev = "54d873ffe99de865ca34e6bb3b92736e29e01619";
  }) { };

    };
  };
};

nixpkgs = builtins.fetchTarball {
    url = "https://github.com/NixOS/nixpkgs/archive/20.03.tar.gz";
    sha256 = "0182ys095dfx02vl2a20j1hz92dx3mfgz2a6fhn31bqlp1wa8hlq";
};

in

{ pkgs ? import nixpkgs { overlays = [ overlay1 myHaskellPackageOverlay ]; }, doBenchmark ? false }:

let

  haskellDeps = ps: with ps; [
    hmatrix
    hmatrix-sundials1
    Naperian
    numbers
  ];

in

  pkgs.stdenv.mkDerivation {
  name = "env";
  buildInputs = [
    (pkgs.myHaskellPackages.ghcWithPackages haskellDeps)
    pkgs.openmpi
    pkgs.openssh
    pkgs.sundials1
    pkgs.python3
    (pkgs.python3.withPackages (ps: [ ps.numpy ps.matplotlib ]))
  ];
}
