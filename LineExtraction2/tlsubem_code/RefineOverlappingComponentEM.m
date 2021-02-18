function [ fineResult ] = RefineOverlappingComponentEM( CCsL,CCsNum,linesL,linesNum,charHeight,options)

% CCsL=L;
% CCsNum=num;
% linesL=Lines;
% linesNum=numLines;

sz = size(CCsL);
fineResult = zeros(sz);

res = zeros(CCsNum,linesNum);
CCsLF = CCsL(:);
linesLF = linesL(:);

for i=1:length(CCsLF)
    if (CCsLF(i) && linesLF(i))
        res(CCsLF(i),linesLF(i)) = 1;
    end
end

temp = sum(res,2);
CCindices = find(temp > 1);
skel = bwmorph(linesL,'skel', Inf);

%handling overlapping components
for i=1:length(CCindices)
    idx = CCindices(i);
    cc = (CCsL == idx);
    linesIndices =  find(res(idx,:));
    
    %Checking if it is really an overlapping component or just an
    %ascender/descender
    if (length(linesIndices) == 2)
        skelLabels = bwlabel(skel & ismember(linesL, linesIndices));
        temp = imreconstruct(cc & skelLabels,skelLabels>0);
        [~,num] = bwlabel(temp);
        if (num < 2)
            continue;
        end
    end
    clear skelLabels;
    clear temp;
    disp(i);
    
    ccPixels = regionprops(cc,'PixelList','PixelIdxList');
    ccPixelList = ccPixels.PixelList;
    ccPixelIdxList =  ccPixels.PixelIdxList;
    
    Dist = zeros(length(ccPixelList),length(linesIndices));
    if length(ccPixelList)>35000
        for j=1:length(linesIndices)
            line = (linesL == linesIndices(j));
            linePixelList = regionprops(line,'PixelList');
            linePixelList = linePixelList.PixelList;
            [~,Dist(:,j)] = knnsearch(linePixelList,ccPixelList);
        end
        [~,loc] =  min(Dist,[],2);
        for j=1:length(linesIndices)
            indices = loc == j;
            fineResult(ccPixelIdxList(indices)) = linesIndices(j);
        end
    else
        numLines=length(linesIndices);
        num=length(ccPixelList);
        CCsparsePixelNs=computePixelNsSystem(ccPixelList,num,options);
        tempLines=ismember(linesL,linesIndices);
        Dc  = computePixelsDC(tempLines,numLines,ccPixelList,num, charHeight);
        [fineLabels] = SplitEm(numLines, num, CCsparsePixelNs,Dc);
        
        uniqueLabels=unique(fineLabels);
        for k=1:length(uniqueLabels)
            fineIndices = fineLabels == uniqueLabels(k);
            for m=1:length(linesIndices)
                line = (linesL == linesIndices(m));
                linePixelIdxList = regionprops(line,'PixelIdxList');
                linePixelIdxList=linePixelIdxList.PixelIdxList;
                if (length(intersect(ccPixelIdxList(fineIndices),linePixelIdxList))>10)
                    fineResult(ccPixelIdxList(fineIndices)) = linesIndices(m);                    
                end
            end
        end
    end
end
end
