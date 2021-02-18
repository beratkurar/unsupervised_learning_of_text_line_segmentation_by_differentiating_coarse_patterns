function [ result ] = mocManualBlobDatasetLinesExtraction(I, bin, varargin)
    if (nargin == 2)

        options = struct('EuclideanDist',true, 'mergeLines', true, 'EMEstimation',false,  'skew',false,...
            'cacheIntermediateResults', false, 'thsLow',10,'thsHigh',200,'Margins', 0.03);
    else
        options = varargin{1};
    end

    charRange=estimateCharsHeight(I,bin,options);

    [L,num] = bwlabel(bin);
    rgblinesMask = imread([options.blobsPath,options.sampleName,'.png']);
    linesMask=imbinarize(rgblinesMask,0.5);
    linesMask=linesMask(:,:,1);

    [LabeledLines,LabeledLinesNum] = bwlabel(linesMask);
    [result ] = moc_manual_blobs_PostProcessByMRF(L, num, LabeledLines, LabeledLinesNum, charRange, options);
end