function [newLines] = MrcnnMergeBlobsExtractLines(rgblinesMask)

graylinesMask=rgblinesMask(:,:,1);
[rows,cols]=size(graylinesMask);
newLines=zeros(rows,cols);
colors=unique(graylinesMask);
for color=2:length(colors)
    tempMask=graylinesMask==colors(color);
    cleanTempMask=bwareaopen(tempMask,1000);
    [~,num]=bwlabel(cleanTempMask);
    if num>=2
        joinedTempMask=MrcnnJoinSegments(cleanTempMask);
        newLines(joinedTempMask==1)=color;
    else
        newLines(cleanTempMask==1)=color;
    end


end




end
