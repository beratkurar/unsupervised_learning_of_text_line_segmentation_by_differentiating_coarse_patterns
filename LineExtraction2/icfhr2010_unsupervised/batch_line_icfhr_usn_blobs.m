close all;


srcPath = 'icfhr2010_unsupervised/icfhr2010_test_binary_images/';
clnPath = 'icfhr2010_unsupervised/icfhr2010_test_inverse_binary_images/';
dstPath = 'icfhr2010_unsupervised/icfhr2010_blob_line_100x150_result/';
blobsPath='icfhr2010_unsupervised/icfhr2010_blob_line_100x150/';

% Evaluation Map estimation -  turn this option on for highly degraded gray scale images.
options = struct('EuclideanDist',true, 'mergeLines', true, 'EMEstimation',false,... 
    'cacheIntermediateResults', true,'blobsPath',blobsPath, 'srcPath',srcPath, 'dstPath', dstPath, 'thsLow',15,'thsHigh',Inf,'Margins', 0);
%options = merge_options(options,varargin{:});
samplesDir = dir(srcPath);
mkdir([dstPath,'fused_polygons']); mkdir([dstPath,'polygon_labels/']);
mkdir([dstPath,'pixel_labels']);
for sampleInd = 1:length(samplesDir)
    fileName = samplesDir(sampleInd).name;
    [path,sampleName,ext] = fileparts(fileName);
    if (strcmp(ext,'.png'))
        options.sampleName = sampleName;
        options.fileName = fileName;
        I = imread( [srcPath,'/',fileName]);
        bin = imread( [clnPath,'/',sampleName,'.png']);
        bin=bin(:,:,1);
        [result,Labels, linesMask, newLines] = FcnBlobsExtractLines(I, bin, options);
        [polygon_labels] = postProcessByBoundPolygonAndPixelsDiva( result);
        DivaSaveResults2Files(I,polygon_labels,result,fileName,dstPath);
    end
end