function [result] = BlobsTouchSplitEmExtractLines(I, bin, varargin)
    if (nargin == 2)
        options = struct('EuclideanDist',true, 'mergeLines', true, 'EMEstimation',false,... 
            'cacheIntermediateResults', false, 'thsLow',15,'thsHigh',Inf,'Margins', 0);
    else
        options = varargin{1};
    end
    charRange=estimateCharsHeight(I,bin,options);
    if (isnan(charRange(1)))
        charRange=[13,16];
    end
    if (options.cacheIntermediateResults &&...
            exist([options.dstPath,'masks/',options.sampleName,'.png'], 'file') == 2)
        linesMask = imread([dstPath,'masks/',sampleName,'.png']);
    else
        %linesMask = LinesExtraction(I, charRange(1):charRange(2));
        %linesMask = LinesExtraction(~bin, charRange(1):charRange(2));
        rgblinesMask = imread([options.blobsPath,options.sampleName,'.png']);
        linesMask=imbinarize(rgblinesMask,0.5);
        linesMask=linesMask(:,:,1);
    end
    [L,num] = bwlabel(bin);
    
    if (num<=2)
        fprintf('only one component \n')
        result=L;
        %Labels=1;
        %newLines=[];
    else
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
        clear I;
        clear bin;
        RefinedCCs = RefineOverlappingComponentEM(L,num, Lines,numLines,charHeight,options );
        tempMask = RefinedCCs > 0;
        result(tempMask) = RefinedCCs(tempMask);
 
    end
 function result = drawLabels(L,Labels)
    L = uint16(L);
    LUT = zeros(1,65536,'uint16');
    LUT(2:length(Labels)+1) = Labels;
    result = double(intlut(L, LUT));
end
end
