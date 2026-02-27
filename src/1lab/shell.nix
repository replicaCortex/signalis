{
  pkgs ? import <nixpkgs> { },
}:
pkgs.mkShell {
  nativeBuildInputs = [
    pkgs.libGL

    # X11 dependencies
    pkgs.xorg.libX11
    pkgs.xorg.libX11.dev
    pkgs.xorg.libXcursor
    pkgs.xorg.libXi
    pkgs.xorg.libXinerama
    pkgs.xorg.libXrandr

    # Web support (uncomment to enable)
    # pkgs.emscripten
  ];

  # Audio dependencies
  LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
    pkgs.alsa-lib
    pkgs.stdenv.cc.cc.lib.outPath
    pkgs.zlib
    pkgs.portaudio
    pkgs.xorg.libX11
    pkgs.libGL
  ];
}
