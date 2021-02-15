function [polygon_labels] = postProcessByBoundPolygonAndPixelsDiva( result)
[rows,cols]=size(result);
labels=unique(result);
polygon_labels=zeros(rows,cols);
    for label=2:(length(labels))
        selected_label=labels(label);
        temp=(result==selected_label);
        [y, x]=find(temp);
        k=boundary(x,y,0.2);
        mask=poly2mask(x(k),y(k),rows,cols);
        intersect=polygon_labels&mask;
        mask(intersect)=0;
        mask=mask|temp;
        polygon_labels(mask==1)=selected_label;
        
    end
end
