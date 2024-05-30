{ nixpkgs ? import <nixpkgs> { } }:
with nixpkgs.pkgs;
pkgs.mkShell {
  name = "segmentation";
  buildInputs = with pkgs; [
    unzip
    git
    stdenv.cc.cc
  ] ++ (with pkgs.python311Packages; [
    virtualenv
    py-cpuinfo
    typing-extensions
    opencv4
    numpy
    pydot
    graphviz
    pipreqs
  ]);

  shellHook = ''
    export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath [
      pkgs.stdenv.cc.cc
    ]}
    virtualenv .venv
    source .venv/bin/activate
    .venv/bin/pip install tensorflow keras
  '';
}

