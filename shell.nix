{
  pkgs ? import <nixpkgs> { },
}:
pkgs.mkShell {
  nativeBuildInputs = with pkgs; [
    uv
    python312Packages.pyqt5
    libsForQt5.qt5.qtwayland
  ];

  LD_LIBRARY_PATH = "$LD_LIBRARY_PATH:${pkgs.stdenv.cc.cc.lib.outPath}/lib:$LD_LIBRARY_PATH";

}
