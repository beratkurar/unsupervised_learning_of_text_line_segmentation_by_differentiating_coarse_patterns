close all;

%for diva hisdb.
srcPath = 'diva_dataset/crop_cb55/';
clnPath = 'diva_dataset/crop_cb55_clean/';
dstPath = 'diva_dataset/crop_cb55_clean_blobs_30_12_3_touch_split_em_result_10_ct_mean_merge/';
blobsPath='diva_dataset/crop_cb55_baselines_blobs_30_12_3/';

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
    disp(sampleName);
    if (strcmp(ext,'.jpg'))
        options.sampleName = sampleName;
        options.fileName = fileName;
        I = imread( [srcPath,'/',fileName]);
        bin = imread( [clnPath,'/',sampleName,'.png']);
        bin=bin(:,:,1);
        [result] = BlobsTouchSplitEmExtractLines(I, bin, options);
        [polygon_labels] = postProcessByBoundPolygonAndPixelsDiva( result);
        DivaSaveResults2Files(I,polygon_labels,result,fileName,dstPath);
        clear polygon_labels;
        clear result;

    end
end