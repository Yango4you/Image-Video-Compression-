clear;

%% Video Compression 5-1g
scales_video = [0.07, 0.2, 0.4, 0.8, 1.0, 1.5, 2, 3, 4, 4.5];
%scales_video = [0.2];

% iterate over all qScale values
for scaleIdx = 1 : numel(scales_video)
    
    qScale = scales_video(scaleIdx);
    
    % read video sequence into RAM as YCbCr and RGB value
    for i=20:40
        filename = "foreman00" + int2str(i) + ".bmp";
        frame{i-19} = ictRGB2YCbCr(double(imread(filename)));
        frame_rgb{i-19} = double(imread(filename));
    end

    % create Huffman Code for Lena small and encode first frame
    [BinaryTree_still, HuffCode_still, BinCode_still, Codelengths_still] = E_4_Huffman(qScale);
    [frame_rec{1}, PSNR_video(1), bitPerPixel_video(1)] = E_4_Milestone(frame_rgb{1}, qScale, BinaryTree_still, HuffCode_still, BinCode_still, Codelengths_still);
    %subplot(121), imshow(uint8(frame_rgb{1})), title('Original')
    %subplot(122), imshow(uint8(ictYCbCr2RGB(frame_rec{1}))), title('Compressed')

    % encode second frame to train Huffman tabels for vectors and error
    [err_im, MV] = motionEstimation(frame{2}, frame_rec{1});
    zeroRun = IntraEncode_new(err_im, qScale);

    % create Huffman table for vectors
    H = hist(MV(:),1:81); % coefficents can be 1-81
    H = H/sum(H);
    [BinaryTree_vectors, HuffCode_vectors, BinCode_vectors, Codelengths_vectors] = buildHuffman(H);

    % create Huffman table for errors
    H = hist(zeroRun(:),-1000:4000); % EoB is 4000 
    H = H/sum(H);
    [BinaryTree_error, HuffCode_error, BinCode_error, Codelengths_error] = buildHuffman(H);

    % encode and decode all frames except the first one
    for i = 2:length(frame)
        ref_im = frame_rec{i-1};
        im = frame{i};
        
        % encode image
        [err_im, MV] = motionEstimation(im, ref_im);
        zeroRun = IntraEncode_new(err_im, qScale);
        
        % encode two bytestreams with huffman 
        bytestream_MV{i} = enc_huffman_new(MV(:), BinCode_vectors, Codelengths_vectors);
        bytestream_error{i} = enc_huffman_new(zeroRun(:) + 1000 + 1, BinCode_error, Codelengths_error); % add offset of 1001
        
        % decode two bytestreams with huffman 
        dec_MV = reshape(dec_huffman_new(bytestream_MV{i}, BinaryTree_vectors, length(MV(:))), size(MV));
        dec_zeroRun = dec_huffman_new(bytestream_error{i}, BinaryTree_error, length(zeroRun(:))) - 1000 -1; % sub offset of 1001

        % decode image
        frame_rec{i} = IntraDecode_w_Error(dec_zeroRun, dec_MV, ref_im, qScale);
        
        % calculate PSNR and bitrate
        PSNR_video(i) = calcPSNR(frame_rgb{i}, ictYCbCr2RGB(frame_rec{i}));    
        bitPerPixel_video(i) = ((numel(bytestream_MV{i}) + numel(bytestream_error{i})) * 8) / (numel(frame{i})/3);
            
    end

    %subplot(121), imshow(uint8(frame_rgb{20})), title('Original')
    %subplot(122), imshow(uint8(ictYCbCr2RGB(frame_rec{20}))), title('Compressed')

    % calculate mean of PSNR and bitrate
    final_PSNR_video(scaleIdx) = mean(PSNR_video);
    final_bitPerPixel_video(scaleIdx)=mean(bitPerPixel_video);

end

%% Still Image Compression 5-1h

scales_still = [0.15, 0.3, 0.7, 1.0, 1.5, 3, 5, 7, 10];
%scales_still = [0.2];

for scaleIdx = 1 : numel(scales_still)
    qScale = scales_still(scaleIdx);

    % % read video sequence into RAM as YCbCr and RGB value
    for i=20:40
        filename = "foreman00" + int2str(i) + ".bmp";
        frame{i-19} = ictRGB2YCbCr(double(imread(filename)));
        frame_rgb{i-19} = double(imread(filename));
    end

    % create Huffman Code for Lena small and encode first frame
    [BinaryTree_still, HuffCode_still, BinCode_still, Codelengths_still] = E_4_Huffman(qScale);

    for i = 1:length(frame)

        [frame_rec{i}, PSNR_still(i), bitPerPixel_still(i)] = E_4_Milestone(frame_rgb{i}, qScale, BinaryTree_still, HuffCode_still, BinCode_still, Codelengths_still);
                
    end

    %subplot(121), imshow(uint8(frame_rgb{20})), title('Original')
    %subplot(122), imshow(uint8(ictYCbCr2RGB(frame_rec{20}))), title('Compressed')

    % calculate mean of PSNR and bitrate
    final_PSNR_still(scaleIdx) = mean(PSNR_still);
    final_bitPerPixel_still(scaleIdx)=mean(bitPerPixel_still);
end

hold on

% Plot solution values
%load('solution_values4.mat')
load('solution_values5.mat')
plot(bpp_solution, psnr_solution, 'b+-'); % reference curve Chapter 5
%plot(bpp_solution_ch4, psnr_solution_ch4, '--+'); % reference curve Chapter 4

% Plot calculated values
plot(final_bitPerPixel_video, final_PSNR_video, 'r*-');
plot(final_bitPerPixel_still, final_PSNR_still, 'g*-');

% Add Label
xlabel('bpp') 
ylabel('PSNR [db]') 


%% Subfunctions
function yuv = ictRGB2YCbCr(rgb)
% Input         : rgb (Original RGB Image)
% Output        : yuv (YCbCr image after transformation)
% YOUR CODE HERE

yuv(:, :, 1) = 0.299*rgb(:, :, 1) + 0.587*rgb(:, :, 2) + 0.114*rgb(:, :, 3);
yuv(:, :, 2) = -0.169*rgb(:, :, 1) - 0.331*rgb(:, :, 2) + 0.5*rgb(:, :, 3);
yuv(:, :, 3) = 0.5*rgb(:, :, 1) - 0.419*rgb(:, :, 2) - 0.081*rgb(:, :, 3);

end

function rgb = ictYCbCr2RGB(yuv)
% Input         : yuv (Original YCbCr image)
% Output        : rgb (RGB Image after transformation)
% YOUR CODE HERE

rgb(:, :, 1) = yuv(:, :, 1) + 1.402*yuv(:, :, 3);
rgb(:, :, 2) = yuv(:, :, 1) - 0.344*yuv(:, :, 2) - 0.714*yuv(:, :, 3);
rgb(:, :, 3) = yuv(:, :, 1) + 1.772*yuv(:, :, 2);

end

function motion_vectors_indices = SSD(ref_image, image)
%  Input         : ref_image(Reference Image, size: height x width)
%                  image (Current Image, size: height x width)
%
%  Output        : motion_vectors_indices (Motion Vector Indices, size: (height/8) x (width/8) x 1 )

% https://de.mathworks.com/help/images/ref/blockproc.html
motion_vectors_indices = blockproc(image(:,:,1), [8,8], @(block_struct) find_smallest_SSD(block_struct.data, block_struct.location,ref_image(:,:,1)));

end

function index = find_smallest_SSD(block_data, block_location, ref_image)

min_ssd = Inf;
min_ssd_vector = [0,0];

for i=-4:4
    for j=-4:4
        ref_y_start = block_location(1)+j;
        ref_x_start = block_location(2)+i;

        if ref_y_start >= 1 && ref_y_start+7 <= size(ref_image,1) && ref_x_start >= 1 && ref_x_start+7 <= size(ref_image,2)

            ref_block = ref_image(ref_y_start:ref_y_start+7, ref_x_start:ref_x_start+7);
            
            ssd = sum((block_data - ref_block).^2, 'all');
            
            if ssd < min_ssd
                min_ssd = ssd;
                min_ssd_vector = [i,j];
            end
        end
        
    end
end

index = 9*(min_ssd_vector(2)+4) + min_ssd_vector(1)+5;

end 

function rec_image = SSD_rec(ref_image, motion_vectors)
%  Input         : ref_image(Reference Image, YCbCr image)
%                  motion_vectors
%
%  Output        : rec_image (Reconstructed current image, YCbCr image)

rec_image = zeros(size(ref_image));

for i=1:size(motion_vectors, 1)
    for j=1:size(motion_vectors, 2)
        
    vector_x = mod(motion_vectors(i,j),9);
    if vector_x == 0
        vector_x = 4;
    else
        vector_x = vector_x - 5;
    end
    vector_y = (motion_vectors(i,j)-vector_x+4)/9-5;
    
    start_y = 1 +(i-1)*8;
    end_y = start_y + 7;
    start_x = 1 +(j-1)*8;
    end_x = start_x + 7;

   
    rec_image(start_y:end_y, start_x:end_x,:) = ref_image(start_y+vector_y:end_y+vector_y, start_x+vector_x:end_x+vector_x,:);

    end
end


end

function [error, motion_vectors] = motionEstimation(image, ref_image)
    motion_vectors = SSD(ref_image, image);
    rec_image = SSD_rec(ref_image, motion_vectors);
    error = image - rec_image;
end

function dst = IntraEncode_new(image, qScale)
%  Function Name : IntraEncode.m
%  Input         : image (YCbCr RGB Image)
%                  qScale(quantization scale)
%  Output        : dst   (sequences after zero-run encoding, 1xN)
image_ycbcr = image;

% DCT because of speed improvements as block_struct
image_ycbcr = blockproc(image_ycbcr, [8,8], @(block_struct) DCT8x8(block_struct.data));

EoB = 4000;
dst = [];

Y_dst = [];
Cb_dst = [];
Cr_dst = [];

for vertical=1:size(image_ycbcr,1)/8
    for horizontal=1:size(image_ycbcr,2)/8
        % Split into 8x8 block
        vertical_end = vertical*8;
        vertical_start = vertical_end -7;
        horizontal_end = horizontal*8;
        horizontal_start = horizontal_end -7;
        image_block = image_ycbcr(vertical_start:vertical_end, horizontal_start:horizontal_end, :);
        
        % Compress Block
        %dct = DCT8x8(image_block);
        quantization = Quant8x8(image_block, qScale);    
        
        
        for plane=1:size(image_ycbcr,3)
            zigzag = ZigZag8x8(quantization(:,:,plane));
            zerorunenc = ZeroRunEnc_EoB(zigzag, EoB);
            if plane == 1
                Y_dst(end+1:end+length(zerorunenc)) = zerorunenc;
            elseif plane == 2 
                Cb_dst(end+1:end+length(zerorunenc)) = zerorunenc;
            else
                Cr_dst(end+1:end+length(zerorunenc)) = zerorunenc;
           end
        end

    end
end

dst = [Y_dst, Cb_dst, Cr_dst];

end

function dst = IntraDecode_new(image, img_size , qScale)
%  Function Name : IntraDecode.m
%  Input         : image (zero-run encoded image, 1xN)
%                  img_size (original image size)
%                  qScale(quantization scale)
%  Output        : dst   (decoded image YCbCr)

EoB = 4000;

decoded_zerorun = reshape(ZeroRunDec_EoB(image, EoB), 64, [])';
decoded_image = [];

blocks_pre_channel = size(decoded_zerorun, 1)/3;

for i=1:blocks_pre_channel  
    
    block_y = decoded_zerorun(i, :)';
    block_cb = decoded_zerorun(i+blocks_pre_channel, :)';
    block_cr = decoded_zerorun(i+2*blocks_pre_channel, :)';
    
    % DeZigZag
    block_y_dezigzag = DeZigZag8x8(block_y(:));
    block_cb_dezigzag = DeZigZag8x8(block_cb(:));
    block_cr_dezigzag = DeZigZag8x8(block_cr(:));
    
    block(:,:,1) = block_y_dezigzag;
    block(:,:,2) = block_cb_dezigzag;
    block(:,:,3) = block_cr_dezigzag;
    
    % DeQuant
    block_quant = DeQuant8x8(block, qScale);
    
    % IDCT
    %block_idct = IDCT8x8(block_quant);
    
    tmp_vector = block_quant(:);
    decoded_image(i,:)= tmp_vector;
end

cnt=1;
for vertical=1:img_size(1)/8
    for horizontal=1:img_size(2)/8
        vertical_end = vertical*8;
        vertical_start = vertical_end -7;
        horizontal_end = horizontal*8;
        horizontal_start = horizontal_end -7;
        
        dst_ycbcr(vertical_start:vertical_end, horizontal_start:horizontal_end, 1) = reshape(decoded_image(cnt,1:64), [8,8]);
        dst_ycbcr(vertical_start:vertical_end, horizontal_start:horizontal_end, 2) = reshape(decoded_image(cnt,65:128), [8,8]);
        dst_ycbcr(vertical_start:vertical_end, horizontal_start:horizontal_end, 3) = reshape(decoded_image(cnt,129:192), [8,8]);
        cnt = cnt + 1;
    end
end

% IDCT because of speed improvements as block_struct
dst = blockproc(dst_ycbcr, [8,8], @(block_struct) IDCT8x8(block_struct.data));

end

function rec_image = IntraDecode_w_Error(zeroRun, MV, ref_image, qScale)
    
    error = IntraDecode_new(zeroRun, size(ref_image), qScale);
    
    rec_image = SSD_rec(ref_image, MV);
    
    rec_image = rec_image + error;
end

function [BinaryTree, HuffCode, BinCode, Codelengths] = E_4_Huffman(qScale)
    lena_small = double(imread('lena_small.tif'));
    k_small  = IntraEncode(lena_small, qScale);
    H = hist(k_small(:),-1000:4000);
    H = H/sum(H);
    [BinaryTree, HuffCode, BinCode, Codelengths] = buildHuffman(H);
end

function [I_rec, PSNR, bitPerPixel] = E_4_Milestone(Image, qScale, BinaryTree, HuffCode, BinCode, Codelengths)

k = IntraEncode(Image, qScale);

%% use trained table to encode k to get the bytestream
bytestream = enc_huffman_new(k(:) + 1000 + 1, BinCode, Codelengths);
dec_bytestream = dec_huffman_new( bytestream, BinaryTree, length(k(:)));
dec_bytestream_reshape = reshape(dec_bytestream, size(k));
k_rec = double(dec_bytestream_reshape - 1000 - 1);

bitPerPixel = (numel(bytestream)*8) / (numel(Image)/3);

%% image reconstruction
I_rec = IntraDecode_new(k_rec, size(Image),qScale);
PSNR = calcPSNR(Image, ictYCbCr2RGB(I_rec));
 
end
 

%% put all used sub-functions here.
function dst = IntraDecode(image, img_size , qScale)
%  Function Name : IntraDecode.m
%  Input         : image (zero-run encoded image, 1xN)
%                  img_size (original image size)
%                  qScale(quantization scale)
%  Output        : dst   (decoded image)

EoB = 4000;

decoded_zerorun = reshape(ZeroRunDec_EoB(image, EoB), 64, [])';
decoded_image = [];

blocks_pre_channel = size(decoded_zerorun, 1)/3;

for i=1:blocks_pre_channel  
    
    block_y = decoded_zerorun(i, :)';
    block_cb = decoded_zerorun(i+blocks_pre_channel, :)';
    block_cr = decoded_zerorun(i+2*blocks_pre_channel, :)';
    
    % DeZigZag
    block_y_dezigzag = DeZigZag8x8(block_y(:));
    block_cb_dezigzag = DeZigZag8x8(block_cb(:));
    block_cr_dezigzag = DeZigZag8x8(block_cr(:));
    
    block(:,:,1) = block_y_dezigzag;
    block(:,:,2) = block_cb_dezigzag;
    block(:,:,3) = block_cr_dezigzag;
    
    % DeQuant
    block_quant = DeQuant8x8(block, qScale);
    
    % IDCT
    %block_idct = IDCT8x8(block_quant);
    
    tmp_vector = block_quant(:);
    decoded_image(i,:)= tmp_vector;
end

cnt=1;
for vertical=1:img_size(1)/8
    for horizontal=1:img_size(2)/8
        vertical_end = vertical*8;
        vertical_start = vertical_end -7;
        horizontal_end = horizontal*8;
        horizontal_start = horizontal_end -7;
        
        dst_ycbcr(vertical_start:vertical_end, horizontal_start:horizontal_end, 1) = reshape(decoded_image(cnt,1:64), [8,8]);
        dst_ycbcr(vertical_start:vertical_end, horizontal_start:horizontal_end, 2) = reshape(decoded_image(cnt,65:128), [8,8]);
        dst_ycbcr(vertical_start:vertical_end, horizontal_start:horizontal_end, 3) = reshape(decoded_image(cnt,129:192), [8,8]);
        cnt = cnt + 1;
    end
end

% IDCT because of speed improvements as block_struct
dst = round(ictYCbCr2RGB(blockproc(dst_ycbcr, [8,8], @(block_struct) IDCT8x8(block_struct.data))));

end

function dst = IntraEncode(image, qScale)
%  Function Name : IntraEncode.m
%  Input         : image (Original RGB Image)
%                  qScale(quantization scale)
%  Output        : dst   (sequences after zero-run encoding, 1xN)
image_ycbcr = ictRGB2YCbCr(image);

% DCT because of speed improvements as block_struct
image_ycbcr = blockproc(image_ycbcr, [8,8], @(block_struct) DCT8x8(block_struct.data));

EoB = 4000;
dst = [];

Y_dst = [];
Cb_dst = [];
Cr_dst = [];

for vertical=1:size(image_ycbcr,1)/8
    for horizontal=1:size(image_ycbcr,2)/8
        % Split into 8x8 block
        vertical_end = vertical*8;
        vertical_start = vertical_end -7;
        horizontal_end = horizontal*8;
        horizontal_start = horizontal_end -7;
        image_block = image_ycbcr(vertical_start:vertical_end, horizontal_start:horizontal_end, :);
        
        % Compress Block
        %dct = DCT8x8(image_block);
        quantization = Quant8x8(image_block, qScale);    
        
        
        for plane=1:size(image_ycbcr,3)
            zigzag = ZigZag8x8(quantization(:,:,plane));
            zerorunenc = ZeroRunEnc_EoB(zigzag, EoB);
            if plane == 1
                Y_dst(end+1:end+length(zerorunenc)) = zerorunenc;
            elseif plane == 2 
                Cb_dst(end+1:end+length(zerorunenc)) = zerorunenc;
            else
                Cr_dst(end+1:end+length(zerorunenc)) = zerorunenc;
           end
        end

    end
end

dst = [Y_dst, Cb_dst, Cr_dst];

end

function coeff = DCT8x8(block)
%  Input         : block    (Original Image block, 8x8x3)
%
%  Output        : coeff    (DCT coefficients after transformation, 8x8x3)


for i=1:size(block,3)
    % faster alternative
    coeff(:,:,i) = dct2(block(:,:,i));   
    
%     % 1d dct of each colum
%     for j=1:size(block,1)
%         coeff(j,:,i)=dct(block(j,:,i));  
%     end
%     
%     % 1d dct of each row
%     for j=1:size(block,2)
%         coeff(:,j,i)=dct(coeff(:,j,i));
%     end
    
end

end

function block = IDCT8x8(coeff)
%  Function Name : IDCT8x8.m
%  Input         : coeff (DCT Coefficients) 8*8*3
%  Output        : block (original image block) 8*8*3


for i=1:size(coeff,3)
    % faster alternative
    block(:,:,i) = idct2(coeff(:,:,i));   
    
%     % 1d idct of each colum
%     for j=1:size(coeff,1)
%         block(j,:,i)=idct(coeff(j,:,i));  
%     end
%     
%     % 1d idct of each row
%     for j=1:size(coeff,2)
%         block(:,j,i)=idct(block(:,j,i));
%     end
    
end

end

function quant = Quant8x8(dct_block, qScale)
%  Input         : dct_block (Original Coefficients, 8x8x3)
%                  qScale (Quantization Parameter, scalar)
%
%  Output        : quant (Quantized Coefficients, 8x8x3)

    L = qScale*[16 11 10 16 24 40 51 61;...
        12 12 14 19 26 58 60 55;...
        14 13 16 24 40 57 69 56;...
        14 17 22 29 51 87 80 62;...
        18 55 37 56 68 109 103 77;...
        24 35 55 64 81 104 113 92;...
        49 64 78 87 103 121 120 101;...
        72 92 95 98 112 100 103 99];
    
    C = qScale*[17 18 24 47 99 99 99 99;...
        18 21 26 66 99 99 99 99;...
        24 13 56 99 99 99 99 99;...
        47 66 99 99 99 99 99 99;...
        99 99 99 99 99 99 99 99;...
        99 99 99 99 99 99 99 99;...
        99 99 99 99 99 99 99 99;...
        99 99 99 99 99 99 99 99];

    quant(:, :, 1) = round(dct_block(:,:,1) ./ L);
    quant(:, :, 2) = round(dct_block(:,:,2) ./ C);
    quant(:, :, 3) = round(dct_block(:,:,3) ./ C);

end

function dct_block = DeQuant8x8(quant_block, qScale)
%  Function Name : DeQuant8x8.m
%  Input         : quant_block  (Quantized Block, 8x8x3)
%                  qScale       (Quantization Parameter, scalar)
%
%  Output        : dct_block    (Dequantized DCT coefficients, 8x8x3)

    L = qScale*[16 11 10 16 24 40 51 61;...
        12 12 14 19 26 58 60 55;...
        14 13 16 24 40 57 69 56;...
        14 17 22 29 51 87 80 62;...
        18 55 37 56 68 109 103 77;...
        24 35 55 64 81 104 113 92;...
        49 64 78 87 103 121 120 101;...
        72 92 95 98 112 100 103 99];
    
    C = qScale*[17 18 24 47 99 99 99 99;...
        18 21 26 66 99 99 99 99;...
        24 13 56 99 99 99 99 99;...
        47 66 99 99 99 99 99 99;...
        99 99 99 99 99 99 99 99;...
        99 99 99 99 99 99 99 99;...
        99 99 99 99 99 99 99 99;...
        99 99 99 99 99 99 99 99];

    dct_block(:, :, 1) = round(quant_block(:,:,1) .* L);
    dct_block(:, :, 2) = round(quant_block(:,:,2) .* C);
    dct_block(:, :, 3) = round(quant_block(:,:,3) .* C);
    
end

function zz = ZigZag8x8(quant)
%  Input         : quant (Quantized Coefficients, 8x8x3)
%
%  Output        : zz (zig-zag scaned Coefficients, 64x3)

ZigZag = [1     2     6     7     15   16   28   29;
          3     5     8     14   17   27   30   43;
          4     9     13   18   26   31   42   44;
          10    12   19   25   32   41   45   54;
          11    20   24   33   40   46   53   55;
          21    23   34   39   47   52   56   61;
          22    35   38   48   51   57   60   62;
          36    37   49   50   58   59   63   64];

zz=zeros(64,size(quant,3));

for i=1:size(quant,3)
    quant_channel=quant(:,:,i);
    zz( ZigZag(:),i ) = quant_channel(:);
end

end

function coeffs = DeZigZag8x8(zz)
%  Function Name : DeZigZag8x8.m
%  Input         : zz    (Coefficients in zig-zag order)
%
%  Output        : coeffs(DCT coefficients in original order)

ZigZag = [1     2     6     7     15   16   28   29;
          3     5     8     14   17   27   30   43;
          4     9     13   18   26   31   42   44;
          10    12   19   25   32   41   45   54;
          11    20   24   33   40   46   53   55;
          21    23   34   39   47   52   56   61;
          22    35   38   48   51   57   60   62;
          36    37   49   50   58   59   63   64];

coeffs=zeros(8,8,size(zz,2));

for i=1:size(zz,2)
    zz_channel = zz(:,i);
    coeffs(:,:,i) = reshape(zz_channel(ZigZag(:))',[8,8]);
end

end

function zze = ZeroRunEnc_EoB(zz, EOB)
%  Input         : zz (Zig-zag scanned sequence, 1xN)
%                  EOB (End Of Block symbol, scalar)
%
%  Output        : zze (zero-run-level encoded sequence, 1xM)

zze = [];

slice_start = 1;

for slice = 64:64:length(zz)
    
    zz_slice = zz(slice_start:slice);
    slice_start = slice + 1;
    
    zero_seq = 0;
    cnt = 0;

    for i = 1:length(zz_slice)

        if zz_slice(i) == 0
            if zero_seq == 1
                cnt = cnt + 1;
            else
                cnt = 0;
                zero_seq = 1;
            end
            if i == length(zz_slice)
                zze(end+1) = EOB;
            end   
        else
            if zero_seq == 1
                zze(end+1) = 0;
                zze(end+1) = cnt;
            end
                zero_seq = 0;
                zze(end+1) = zz_slice(i); 
        end
    end
end
end

function dst = ZeroRunDec_EoB(src, EoB)
%  Function Name : ZeroRunDec1.m zero run level decoder
%  Input         : src (zero run encoded sequence 1xM with EoB signs)
%                  EoB (end of block sign)
%
%  Output        : dst (reconstructed zig-zag scanned sequence 1xN)

dst = [];
block_cnt = 0;
i = 1;
    
while i <= length(src)

    if src(i) == 0
        dst(end+1:end+1+src(i+1)) = 0;
        block_cnt = block_cnt + src(i+1)+1;
        i = i + 2;
    elseif src(i) == EoB
        dst(end+1:end+(64-block_cnt)) = 0;
        i = i +1;
        block_cnt = 64;
    else
        dst(end+1) = src(i);
        i = i + 1;
        block_cnt = block_cnt + 1;
    end

    if block_cnt == 64
        block_cnt = 0;
    end
end

end

function MSE = calcMSE(Image, recImage)
% Input         : Image    (Original Image)
%                 recImage (Reconstructed Image)
% Output        : MSE      (Mean Squared Error)
[W,H,C] = size(recImage);
error_sum = 0.0;
for i = 1:W
    for j = 1:H
        for k = 1:C
            Y = double(Image(i,j,k));
            Y_dash = double(recImage(i,j,k));
            error_value = (Y - Y_dash)^2;
            error_sum = error_sum + double(error_value);
        end
    end
end
MSE = 1/(W*H*C)*error_sum;
end

function PSNR = calcPSNR(Image, recImage)
% Input         : Image    (Original Image)
%                 recImage (Reconstructed Image)
%
% Output        : PSNR     (Peak Signal to Noise Ratio)
MSE = calcMSE(Image, recImage);
PSNR = 10*log10((2^8-1)^2/MSE);
end
