close all;


srcPath = 'diva_dataset/crop_csg863_clean_inverse/';
clnPath = 'diva_dataset/crop_csg863_clean/';
dstPath = 'diva_unsupervised/csg863_usn_blobs_result/';
blobsPath='diva_unsupervised/csg863_usn_blobs/';

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