close all;

%for diva hisdb.
srcPath = 'diva_dataset/crop_csg863/';
clnPath = 'diva_dataset/crop_csg863_clean/';
dstPath = 'diva_dataset/crop_csg863_merged_mrcnn_post_processed_polygon_labels/';
blobsPath='diva_dataset/crop_csg863_mrcnn_post_processed_polygon_labels/';

% Evaluation Map estimation -  turn this option on for highly degraded gray scale images.
options = struct('EuclideanDist',true, 'mergeLines', true, 'EMEstimation',false,... 
    'cacheIntermediateResults', true,'blobsPath',blobsPath, 'srcPath',srcPath, 'dstPath', dstPath, 'thsLow',15,'thsHigh',Inf,'Margins', 0);
%options = merge_options(options,varargin{:});
samplesDir = dir(srcPath);
mkdir([dstPath,'merged_polygons/']);
mkdir([dstPath,'fused_polygons']);
for sampleInd = 2:length(samplesDir)
    fileName = samplesDir(sampleInd).name;
    [path,sampleName,ext] = fileparts(fileName);
    if (strcmp(ext,'.jpg'))
        I = imread( [srcPath,'/',fileName]);
        rgblinesMask = imread([blobsPath,sampleName,'.png']);
        newLines = MrcnnMergeBlobsExtractLines(rgblinesMask);
        imwrite(uint8(newLines),[dstPath,'/merged_polygons/',sampleName,'.png']);
        blended = imfuse(I,label2rgb(newLines),'blend');
        imwrite(blended, [dstPath,'/fused_polygons/',sampleName,'.png']);
    end
end