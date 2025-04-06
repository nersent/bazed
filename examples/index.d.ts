declare global {
  declare const {
    execute,
    env,
    fileSet,
    glob,
    maturin,
    nextJs,
    nodeBinary,
    pythonBinary,
    nodeMainWrapper,
    pythonLibrary,
    rustLibrary,
    tsLibrary,
    writeFile,
  }: any;
}
