window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    // So restrinja o MathJax aos spans/divs que o pymdownx.arithmatex marca com
    // class="arithmatex" - sem isto, o re-typeset abaixo pode disputar com o
    // typeset automatico do MathJax na primeira carga da pagina.
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};

// Re-typeset a cada navegacao (Material usa carregamento instantaneo via document$,
// que dispara inclusive na primeira carga da pagina). Sem limpar o estado anterior
// (clearCache/typesetClear/texReset), esse primeiro disparo corre junto com o
// typeset automatico do MathJax e pode apagar a formula sem renderizar nada no
// lugar - por isso os tres resets vêm antes do typesetPromise.
document$.subscribe(() => {
  MathJax.startup.output.clearCache();
  MathJax.typesetClear();
  MathJax.texReset();
  MathJax.typesetPromise();
});
