{
  nixpkgs ? import <nixpkgs> {
    config.allowUnfree = true; #config.cudaSupport = true;
  },
  pkgs ? nixpkgs.pkgs
}:
let

  namex = with pkgs; with pkgs.python3.pkgs; buildPythonPackage rec {
    pname = "namex";
    version = "0.0.8";
    format = "pyproject";

    disabled = pythonOlder "3.9";

    src = fetchPypi {
      inherit pname version;
      hash = "sha256-MqUPbFZcC7EKp2KYyVlQer3A6FDv4IXcOPNED8s6qQs=";
    };

    inputsEnv = python3.withPackages (p: with p; [
      setuptools
    ]);

    nativeBuildInputs = [
      inputsEnv
    ];
  };

  keras = with pkgs; with pkgs.python3.pkgs; buildPythonPackage rec {
    pname = "keras";
    version = "3.3.3";
    format = "pyproject";

    disabled = pythonOlder "3.9";

    src = fetchPypi {
      inherit pname version;
      hash = "sha256-8v3/yENP13BFz4+yGBbbqiMI1fdpdMqSSy9gtAQzsaA=";
    };

    inputsEnv = python3.withPackages (p: with p; [
      setuptools
      numpy
      absl-py
      rich
      h5py
      optree
      ml-dtypes
      namex
    ]);

    nativeBuildInputs = [
      inputsEnv
    ];
  };

  tensorflow = with pkgs; with pkgs.python3.pkgs; buildPythonPackage rec {
    pname = "tensorflow";
    version = "2.16.1";
    format = "wheel";

    disabled = pythonOlder "3.9";

    src = fetchPypi rec {
      inherit pname version format;
      hash = "sha256-kwxhEAzOOly2PTD+Z3ZQRAUhToOYomypaCIuy4uPlAQ=";
      dist = python;
      python = abi;
      abi = "cp311";
      platform = "manylinux_2_17_x86_64.manylinux2014_x86_64";
    };

    inputsEnv = python3.withPackages (p: with p; [
      setuptools
      numpy
    ]);

    nativeBuildInputs = [
      inputsEnv
    ];
  };

  ultralytics-thop = with pkgs; with pkgs.python3.pkgs; buildPythonPackage rec {
    pname = "ultralytics_thop";
    version = "0.2.7";
    format = "pyproject";

    disabled = pythonOlder "3.7";

    src = fetchPypi {
      inherit pname version;
      hash = "sha256-YPYcr38nxHwNGOL87nrwFPqCtXpnp00+OlSInImubZs=";
    };

    inputsEnv = python3.withPackages (p: with p; [
      setuptools
      numpy
      opencv4
      torch
    ]);

    nativeBuildInputs = [
      inputsEnv
    ];
  };

  ultralytics = with pkgs; with pkgs.python3.pkgs; buildPythonPackage rec {
    pname = "ultralytics";
    version = "8.2.28";
    format = "pyproject";

    disabled = pythonOlder "3.7";

    src = fetchPypi {
      inherit pname version;
      hash = "sha256-cAvaOWC1k0ggWhKm+biT0f1zy7yFQT2r5Q8bs/PgMGY=";
    };

    doCheck = false;

    pythonRemoveDeps = [
      "opencv-python"
      "torchvision"
    ];

    prePatch = ''
      export HOME=$(mktemp -d)
    '';

    inputsEnv = python3.withPackages (p: with p; [
      setuptools
      numpy
      torch
      matplotlib
      opencv4
      pillow
      pyyaml
      requests
      scipy
      torch
      torchvision
      tqdm
      psutil
      py-cpuinfo
      pandas
      seaborn
    ]);

    nativeBuildInputs = [
      pkgs.python3.pkgs.pythonRelaxDepsHook
      inputsEnv
      ultralytics-thop
    ];
  };

  pythonEnv = pkgs.python3.withPackages (p: with p; [
    ipykernel
    py-cpuinfo
    typing-extensions
    opencv4
    numpy
    pydot
    graphviz
    gradient 
    torch
    torchvision
    matplotlib
    dill
    pandas
    optree
    rich
    ml-dtypes
    wrapt
    gast
    opt-einsum
    flatbuffers
    h5py
    (lib.hiPrio tensorboard)
    (lib.hiPrio tensorboard-plugin-profile)
    (lib.hiPrio protobuf)
    (lib.hiPrio grpcio)
    tensorflow
    keras
    ultralytics
    ultralytics-thop
  ]);

  buildInputs = with pkgs; [
    python312Packages.venvShellHook
    autoPatchelfHook
    unzip
    pipreqs
    git
    stdenv.cc.cc
    pythonEnv
  ];

  venvDir = "./.venv";
in
pkgs.mkShell {
  name = "colorization";

  inherit buildInputs venvDir;

  LANG = "en_US.UTF-8";

  shellHook = ''
    export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath [
      pkgs.stdenv.cc.cc
    ]}

    if [[ -d "${venvDir}" ]]; then
      echo "Skipping venv creation, '${venvDir}' already exists"
      source "${venvDir}/bin/activate"
    else
      echo "Creating new venv environment in path: '${venvDir}'"
      python -m venv "${venvDir}"

      find ${pythonEnv}/bin -mindepth 1 -maxdepth 1 -not -name python | while read i
      do
          ln -s $i ${venvDir}/bin
      done

      find ${pythonEnv}/include -mindepth 1 -maxdepth 1 | while read i
      do
          ln -s $i ${venvDir}/include
      done

      find ${pythonEnv}/lib -mindepth 1 -maxdepth 1 | while read i
      do
          ln -s $i ${venvDir}/lib
      done

      source "${venvDir}/bin/activate"
    fi

    export SITE=${pythonEnv}

    # Gradiant API KEY is in .env
    source .env
  '';
}

