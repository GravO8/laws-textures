1. Rotational Invariance - Perceber como detetar quais vetores quando multiplicados resultam em kernels simétricos de maneira eficiente, i.e. em vez de ter que os gerar a todos e comparar 1 a 1 quais são iguais ao simétrico do outro.

u^T . v == (v^T . u).T é sempre verdade, para qualquer u e v
Fazendo a matriz de todas as N^2 combinations dos vetores, mantemos os kernels que ficam na diagonal e os triangulos sao averaged num so (porque os seus respeitvos kernels são simétricos). O primeiro kernel LxLx é descartado porque esse kernel não tem a 0 sum property: 
*The table of F-ratios shows that it performs poorly only with L3L3, the 3x3 operator that is not zero-sum.* - page 116

2. Normalization - Perceber qual é exatamente o pre processing que se faz antes de aplicar os laws filters

Maybe this:?
<!-- smooth = ones(averWindSize, averWindSize)/(averWindSize^2);
imageG=imfilter(imageG,smooth,'conv','symmetric'); -->



3. Perceber como ir a partir dos convolution maps até números apenas

Como na tabela 7.1. de https://courses.cs.washington.edu/courses/cse576/book/ch7.pdf

*The final output is a segmented image or classification map. Classification is simple and fast if the texture classes are known a priori. Fither texture energy planes or principal component planes may be used as input to the pixel classifier. Clustering or segmentation algorithms must be used if texture classes are unknown.* - page 144/195



4. Gerar os larger vector sets a partir dos basis vectors originais:
 - L3 = [ 1, 2, 1]
 - E3 = [-1, 0, 1]
 - S3 = [-1, 2,-1]
 
 Perceber como é que eles podem ser combinados sem repetições


# Presentation

## What are Laws Textures?
 - radiomic features
 - separable filters
 - generalazible for higher dimensions and larger kernels
 
## Examples of the generated filters

## Why is it useful?
 - Bar code segmentation
 - 3D dataset classification
 
## Code on GitHub
https://github.com/GravO8/laws-textures
