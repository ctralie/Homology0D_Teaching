function [] = makeHairMasks(directoryName)
    files = dir(sprintf('%s/*.jpg', directoryName));
    for ii = 1:length(files)
        name = files(ii).name;
        path = sprintf("%s/%s", directoryName, name);
        [filepath, name, ext] = fileparts(name);
        maskpath=  sprintf("%s/%s_mask.png", directoryName, name);
        I = imread(path);
        mask = roipoly(I);
        imwrite(mask, maskpath);
    end
end

