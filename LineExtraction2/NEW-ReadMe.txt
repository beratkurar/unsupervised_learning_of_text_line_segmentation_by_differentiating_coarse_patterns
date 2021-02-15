1. Set path to all the folders:
-anigauss
-Binarization
-Code
-EvolutionMap
-gco-v3.0
-matlab-bgl
-Multi_Skew_Code
-SLMtools

2. Run gco-v3.0\matlab\GCO_UnitTest.m

3. Run either one:

I = imread('101.tif');
bin = ~I;		  						% ICDAR is composed of binary images. We assume that the text is brigher than the background.	
[result,Labels, linesMask, newLines] = ExtractLines(I, bin);		% Extract the lines, linesMask = intermediate line results for debugging.
imshow(label2rgb(result))						% Display result


The code for multi-skew lines is run using the following commands:

I = imread('ms_25.png');
bin = ~I;
[ result,Labels, finalLines, newLines, oldLines ] = multiSkewLinesExtraction(I, bin);
imshow(label2rgb(result))