close all;
imgPath = 'ahte_dataset/ahte_train_binary_images/';
srcPath = 'ahte_dataset/ahte_train_pixel_label/';
dst2Path = 'ahte_dataset/ahte_train_rgb_polygon_label/';
dst1Path = 'ahte_dataset/ahte_train_polygon_label/';

samplesDir = dir(imgPath);
mkdir(dst1Path);
mkdir(dst2Path);
for sampleInd = 1:length(samplesDir)
    fileName = samplesDir(sampleInd).name;
    if (~strcmp(fileName(1), '.')  && ~samplesDir(sampleInd).isdir)
        fileName = samplesDir(sampleInd).name;
        [path,sampleName,ext] = fileparts(fileName);
        fprintf('%d - filename %s \n',sampleInd,fileName);
        I = imread( [imgPath,fileName]);
        pixel_labels = imread( [srcPath,sampleName,'.png']);
        [polygon_labels] = MrcnnPostProcessByThreshedBoundPolygon(pixel_labels);     
        imwrite(uint8(polygon_labels),[dst1Path,fileName]);
        blended = imfuse(I,label2rgb(polygon_labels),'blend');
        imwrite(blended,[dst2Path,fileName]);
    end
end





