function [charsRange] = estimateCharsHeight(I,bin,options)
% Estimating line heights.

% Binary Image.
if (islogical(I))
    % Assuming ICDAR's file format
    [lower,upper] = estimateBinaryHeight(bin,options.thsLow,options.thsHigh,options.Margins);    
    
    %Gray Scale or Color image.
else    
    if (options.EMEstimation)
        if (options.EMEstimation)
            EvolutionMapDirectory = 'D:\Dropbox\PhD\ISF2010\Code\LineExtraction\EvolutionMap\';
            EMResult = [EvolutionMapDirectory, 'estimateCharacterDimensions.exe',' ','"',options.srcPath,...
                '/',options.fileName,'"'];
            [~,cmdout] = system(EMResult);
            res = sscanf(cmdout,'(%f,%f)\n(%f,%f)');
        end
    end
    
    % Evolution map is turned off or failed to execute.
    if (~options.EMEstimation || isempty(res) || res(3) == 0)
        [lower,upper] = estimateBinaryHeight(bin,options.thsLow,options.thsHigh,options.Margins);
    else
        % Success.
        lower = res(3)/2;
        upper = res(4)/2;
    end
end

charsRange = [lower,upper];
end
