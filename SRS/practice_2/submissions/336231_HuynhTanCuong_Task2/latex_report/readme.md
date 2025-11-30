This is an English-language template for students' essays and synopses based on GOST 7.32-2017. Originally prepared in accordance with the recommendations from the academic and teaching staff of ITMO University for students of technical and natural sciences.

The template is created for XeLaTeX compiler; the latter has some features (e.g., native support of Unicode) that are abcent in some older compilers like pdfLaTeX. Therefore, do not be surprised if nothing works upon your attempt to compile with pdfLaTeX - it won't. All or almost all of the current LaTeX distributions already contain XeLaTeX, just select it in Settings.

Bibliography is assembled by biber. If compilation ends up with an error or you get an undesired result in terms of formatting, please check if biber is selected as the processor of choice for bibliography in your LaTeX setup. In Overleaf, everything works by default. In some other setups, it may be necessary to choose the bibliography processor manually in Settings (e.g., TeXstudio tries to load bibTeX by default).

Originally the template was prepared in Overleaf, but it has been also tested on Windows 11 in combination MikTeX + TeXstudio.