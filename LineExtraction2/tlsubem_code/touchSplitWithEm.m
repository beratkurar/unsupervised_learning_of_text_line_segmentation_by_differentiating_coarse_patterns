
options = struct('EuclideanDist',true, 'mergeLines', true, 'EMEstimation',false,... 
    'cacheIntermediateResults', false, 'thsLow',15,'thsHigh',Inf,'Margins', 0);

I = imread('org2.jpg');
bin = imread('binary2.png');
bin=bin(:,:,1);

charRange=estimateCharsHeight(I,bin,options);

rgblinesMask = imread('blob2.png');
linesMask=imbinarize(rgblinesMask,0.5);
linesMask=linesMask(:,:,1);

[L,num] = bwlabel(bin);

CCsparseNs = computeNsSystem( L,num,options);
[Lines, numLines] = permuteLabels(bwlabel(linesMask));
Dc  = computeLinesDC(Lines,numLines,L,num, charRange(2));
[ LabelCost ] = computeLinesLabelCost( L,Lines,numLines );
[Labels] = LineExtraction_GC_MRFminimization(numLines, num, CCsparseNs,Dc, LabelCost);
Labels(Labels == numLines+1) = 0;
residualLines = ismember(Lines, Labels);
Lines(~residualLines)=0;
result = drawLabels(L,Labels);
charHeight=charRange(2);
RefinedCCs = RefineOverlappingComponentEM(L,num, Lines,numLines,charHeight,options );
tempMask = RefinedCCs > 0;
result(tempMask) = RefinedCCs(tempMask);
 
function result = drawLabels(L,Labels)
    L = uint16(L);
    LUT = zeros(1,65536,'uint16');
    LUT(2:length(Labels)+1) = Labels;
    result = double(intlut(L, LUT));
end
