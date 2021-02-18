close all;
srcPath = 'moc_dataset/moc_test_images/';
dstPath = 'moc_dataset/moc_test_fcn_blobs_result_10_ct_mean_merge/';
blobsPath='moc_dataset/moc_test_fcn_blobs/';

% Evaluation Map estimation -  turn this option on for highly degraded gray scale images.
options = struct('EuclideanDist',true, 'mergeLines', false, 'EMEstimation',false, 'skew',false,...
    'cacheIntermediateResults', false,'blobsPath', blobsPath, 'srcPath',srcPath, 'dstPath', dstPath, 'thsLow',10,'thsHigh',200,'Margins', 0.03);


samplesDir = dir(srcPath);
mkdir([dstPath,'fused_polygons']); 
mkdir([dstPath,'polygon_labels']);
mkdir([dstPath,'pixel_labels']);
for sampleInd = 1:length(samplesDir)
    fileName = samplesDir(sampleInd).name;
    display(fileName);
    if (~strcmp(fileName(1), '.')  && ~samplesDir(sampleInd).isdir)
        fileName = samplesDir(sampleInd).name;
        [path,sampleName,ext] = fileparts(fileName);
        sampleNum = str2double(sampleName(8:end));
        I = imread( [srcPath,'/',fileName]);
        bin = ~I;
        options.sampleName = sampleName;
        options.fileName = fileName;

        [result ] = mocManualBlobDatasetLinesExtraction(I, bin, options);
        [polygon_labels] = postProcessByBoundPolygonAndPixelsDiva( result);
        DivaSaveResults2Files(I,polygon_labels,result,fileName,dstPath); close all;
    end
end