clear;
close all;
% this file without functions buildHuffman(), enc_huffman_new(), dec_huffman_new() 

% read lena_small
lena_small = double(imread('./data/images/lena_small.tif'));

% read all images from foreman and change RGB to YCbCr 
for i = 1:21
    imagename= "./sequences/foreman20_40_RGB/foreman00"+int2str(i+19)+".bmp";
    frame_RGB{i} = double (imread(imagename));
    frame_YCbCr{i} = ictRGB2YCbCr(frame_RGB{i});
end

% qScale for still image and video
scales_still = [0.15,0.3,0.7,1.0,1.5,3,5,7,10];
scales_video = [0.07, 0.2, 0.4, 0.8, 1.0, 1.5, 2, 3, 4, 4.5];

% stll image codec
[bpp_still, PSNR_still] = still_image(lena_small, scales_still, frame_RGB, frame_YCbCr);

% video codec
[bpp_video, PSNR_video] = video(lena_small, scales_video, frame_RGB, frame_YCbCr);

% plot
figure(1)
plot(bpp_video, PSNR_video, 'b+-', bpp_still, PSNR_still, 'r+-');
legend('video codec','still image codec');
xlabel('bitPerPixel');
ylabel('PSNR[dB]');

%figure(2)
%load('solution_values5.mat')
%plot(bpp_solution, psnr_solution, '--*', bpp_video, PSNR_video, 'b+-', bpp_still, PSNR_still, 'r+-');
%legend('solution curver', 'video codec curver', 'still image codec');
%xlabel('bitPerPixel');
%ylabel('PSNR[dB]');

% sub-functions
function [bpp_still, PSNR_still] = still_image(lena_small, scales_still, frame_RGB, frame_YCbCr)
    for scaleIdx = 1 : numel(scales_still)
        qScale   = scales_still(scaleIdx);

        % build huffman code with Lena_small
        k_small  = IntraEncode(lena_small, qScale);
        lower_bound = -1000;
        upper_bound = 4000;
        PMF = stats_marg(k_small,lower_bound:upper_bound);
        [BinaryTree_still, HuffCode_still, BinCode_still, Codelengths_still] = buildHuffman(PMF);   

        for i = 1:length(frame_RGB)
            k = IntraEncode(frame_YCbCr{i}, qScale);
            offset= -(lower_bound-1);

            % huffman encoding and decoding of foreman image
            bytestream = enc_huffman_new(k+offset, BinCode_still, Codelengths_still);
            k_rec = double(reshape(dec_huffman_new(bytestream, BinaryTree_still, length(k(:))),size(k)))-offset;

            % reconstruction of foreman image and back to RGB
            I_rec = IntraDecode(k_rec, size(frame_RGB{i}),qScale);
            decoded_frame_still{i} = ictYCbCr2RGB(I_rec); 

            % get PSNR and bit per pixel
            bpp(i) = (numel(bytestream)*8)/(numel(frame_RGB{i})/3);
            PSNR(i) = calcPSNR(frame_RGB{i}, decoded_frame_still{i});
        end

        % get mean of PSNR and bit per pixel
        bpp_still(scaleIdx) = mean(bpp);
        PSNR_still(scaleIdx) = mean(PSNR);
    end
end

function [bpp_video, PSNR_video] = video(lena_small, scales_video, frame_RGB, frame_YCbCr)
    for scaleIdx = 1 : numel(scales_video)
        qScale = scales_video(scaleIdx);

        % build huffman code with Lena_small
        k_small = IntraEncode(lena_small, qScale);
        lower_bound = -1000;
        upper_bound = 4000;
        PMF = stats_marg(k_small,lower_bound:upper_bound);
        [BinaryTree_video, HuffCode_video, BinCode_video, Codelengths_video] = buildHuffman(PMF);

        % encoding and decoding the first image
        frame_first = ictRGB2YCbCr(double(frame_RGB{1}));
        k_first = IntraEncode(frame_first, qScale);
        offset= -(lower_bound-1);
        bytestream = enc_huffman_new(k_first+offset, BinCode_video, Codelengths_video);
        k_rec = double(reshape(dec_huffman_new(bytestream, BinaryTree_video, length(k_first(:))), size(k_first) ))-offset;

        % reconstruction of the first image
        I_rec = IntraDecode(k_rec, size(frame_first),qScale);
        decoded_frame_video{1}= ictYCbCr2RGB(I_rec); 

        % get PSNR and bit per pixel of the first image
        bpp(1) = (numel(bytestream)*8)/(numel(frame_first)/3);
        PSNR(1) = calcPSNR(frame_RGB{1}, decoded_frame_video{1});

        for i = 2:length(frame_RGB)
            ref_im =  ictRGB2YCbCr(decoded_frame_video{i-1});

            % get motion vector
            mv = SSD(ref_im(:,:,1), frame_YCbCr{i}(:,:,1));
            frame_YCbCr_ref = SSD_rec(ref_im, mv);

            % get prediction error
            error = frame_YCbCr{i} - frame_YCbCr_ref;

            %  build huffman code with mv and prediction error
            k = IntraEncode(error,qScale);
            if i == 2
                PMF_mv = stats_marg(mv,  1: 81);
                [BinaryTree_mv, HuffCode_mv, BinCode_mv, Codelengths_mv] = buildHuffman(PMF_mv);
                PMF_predError = stats_marg(k,  lower_bound: upper_bound);
                [BinaryTree_predError, HuffCode_predError, BinCode_predError, Codelengths_predError] = buildHuffman(PMF_predError);
            end

            % huffman encoding and decoding of motion vectors
            bytestream_mv = enc_huffman_new(mv, BinCode_mv, Codelengths_mv);
            mv_rec = double(reshape(dec_huffman_new(bytestream_mv, BinaryTree_mv, length(mv(:))),size(mv)));

            % huffman encoding and decoding of prediction error
            bytestream_predError = enc_huffman_new(k+offset, BinCode_predError, Codelengths_predError);
            predError_rec = double(reshape(dec_huffman_new(bytestream_predError, BinaryTree_predError, length(k(:))),size(k)))-offset; 

            % creating motion compensation and bacr to RGB
            image_rec = motion_comp(predError_rec, mv_rec, ref_im, qScale);
            decoded_frame_video{i} =  ictYCbCr2RGB(image_rec );

            % get PSNR and bit per pixel 
            bpp(i) = ((numel(bytestream_mv)+numel(bytestream_predError))*8)/(numel(frame_RGB{i})/3);
            PSNR(i) = calcPSNR(frame_RGB{i},decoded_frame_video{i});
        end

        % get mean of PSNR and bit per pixel
        bpp_video(scaleIdx) = mean(bpp);
        PSNR_video(scaleIdx) = mean(PSNR);
    end
end

function image_rec = motion_comp(predError_rec, mv_rec, ref_im, qScale)
    mcp_rec = SSD_rec(ref_im, mv_rec); 
    predError_rec = IntraDecode(predError_rec, size(ref_im),qScale);
    image_rec = mcp_rec + predError_rec;
end

function motion_vectors_indices = SSD(ref_image, image)
    [w,h] = size(image);
    indexMatrix = reshape(1:81,9,9);
    ref_image = padarray(ref_image,[4,4],0,'both');
    motion_vectors_indices = [];
    for i = 1:8:h
        for j = 1:8:w
            sseMatrix = image(j:j+7,i:i+7);
            minSSE = 9999999999;
            for k = i:i+8
                for l = j:j+8
                    refMatrix = ref_image(l:l+7, k:k+7);
                    curentSSE = sum(sum((refMatrix - sseMatrix).^2));
                    index_x = k-i+1;
                    index_y = l-j+1;
                    if curentSSE < minSSE
                        minSSE = curentSSE;
                        curentX = index_x;
                        curentY = index_y;
                    end
                end
            end
            motion_vectors_indices((j-1)/8+1, (i-1)/8+1) = indexMatrix(curentX,curentY);
        end
    end
end

function rec_image = SSD_rec(ref_image, motion_vectors)
    [w,h,c] = size(ref_image);    
    for i = 1:8:h
        for j = 1:8:w
                indexY = ceil(motion_vectors( (j-1)/8+1,(i-1)/8+1 )/9)-5;
                if mod(motion_vectors((j-1)/8+1,(i-1)/8+1),9) == 0
                    indexX = 4;
                else
                    indexX = mod(motion_vectors((j-1)/8+1,(i-1)/8+1),9)-5;
                end
                if j+indexY>0 && i+indexX>0 && w>=j+indexY+7 && h>=i+indexX+7
                    rec_image(j:j+7,i:i+7,:) = ref_image(j+indexY:j+indexY+7,i+indexX:i+indexX+7,:);
                else
                    rec_image(j:j+7,i:i+7,:) = ref_image(j:j+7,i:i+7,:);
                end
        end
    end
end

function dst = IntraEncode(image, qScale)
    EoB = 4000;
    dct = blockproc(image, [8, 8], @(block_struct) DCT8x8(block_struct.data));
    quant = blockproc(dct, [8, 8], @(block_struct) Quant8x8(block_struct.data,qScale));
    zze =  blockproc(quant , [8, 8], @(block_struct) ZigZag8x8(block_struct.data)); 
    zze_1 = zze(:,1:3:end);
    zze_2 = zze(:,2:3:end);
    zze_3 = zze(:,3:3:end);
    zze_1 = blockproc(zze_1, [64, size(zze,2)/3], @(block_struct) block_struct.data(:)); 
    zze_2 = blockproc(zze_2, [64, size(zze,2)/3], @(block_struct) block_struct.data(:)); 
    zze_3 = blockproc(zze_3, [64, size(zze,2)/3], @(block_struct) block_struct.data(:));   
    zze = [zze_1,zze_2,zze_3];
    dst = ZeroRunEnc_EoB(zze(:),EoB);
end

function dst = IntraDecode(image, img_size , qScale)
    EoB = 4000;
    zzd = ZeroRunDec_EoB(image,EoB);
    zzd = reshape(zzd,[prod(img_size)/3,3]);
    dec_quant1 =  blockproc(reshape(zzd(:,1), [64*(img_size(1)/8)*(img_size(2)/8),1]),[64*(img_size(2)/8),1], @(block_struct) reshape(block_struct.data,[64,(img_size(2)/8)]));
    dec_quant2 =  blockproc(reshape(zzd(:,2), [64*(img_size(1)/8)*(img_size(2)/8),1]),[64*(img_size(2)/8),1], @(block_struct) reshape(block_struct.data,[64,(img_size(2)/8)]));
    dec_quant3 =  blockproc(reshape(zzd(:,3), [64*(img_size(1)/8)*(img_size(2)/8),1]),[64*(img_size(2)/8),1], @(block_struct) reshape(block_struct.data,[64,(img_size(2)/8)]));
    dec_quant1 =  blockproc(dec_quant1, [64,1], @(block_struct) DeZigZag8x8(block_struct.data));
    dec_quant2 =  blockproc(dec_quant2, [64,1], @(block_struct) DeZigZag8x8(block_struct.data));
    dec_quant3 =  blockproc(dec_quant3, [64,1], @(block_struct) DeZigZag8x8(block_struct.data));
    dec_quant = cat(3,dec_quant1,dec_quant2,dec_quant3);
    dctd = blockproc(dec_quant, [8, 8], @(block_struct) DeQuant8x8(block_struct.data,qScale));
    dst = blockproc(dctd, [8, 8], @(block_struct) IDCT8x8(block_struct.data));
end

function PSNR = calcPSNR(Image, recImage)
    PSNR = 10*log10((2^8-1).^2/calcMSE(Image, recImage));
end

function MSE = calcMSE(Image, recImage)
    MSE=sum((double(Image(:))- double(recImage(:))).^2)/numel(Image);
end

function coeff = DCT8x8(block)
    B=block;
    coeff(:,:,1) = dct2(B(:,:,1));
    coeff(:,:,2) = dct2(B(:,:,2));
    coeff(:,:,3) = dct2(B(:,:,3));
end

function dct_block = DeQuant8x8(quant_block, qScale)
    L = [16 11 10 16 24 40 51 61; 12 12 14 19 26 58 60 55; 14 13 16 24 40 57 69 56; 14 17 22 29 51 87 80 62;
         18 55 37 56 68 109 103 77; 24 35 55 64 81 104 113 92; 49 64 78 87 103 121 120 101; 72 92 95 98 112 100 103 99];
    
    C = [17 18 24 47 99 99 99 99; 18 21 26 66 99 99 99 99; 24 13 56 99 99 99 99 99; 47 66 99 99 99 99 99 99;
         99 99 99 99 99 99 99 99; 99 99 99 99 99 99 99 99; 99 99 99 99 99 99 99 99; 99 99 99 99 99 99 99 99];
    dct_block(:,:,1) = quant_block(:,:,1).*(L*qScale);
    dct_block(:,:,2) = quant_block(:,:,2).*(C*qScale);
    dct_block(:,:,3) = quant_block(:,:,3).*(C*qScale);
end

function coeffs = DeZigZag8x8(zz)
    ZZ = [1 2 6 7 15 16 28 29; 3 5 8 14 17 27 30 43; 4 9 13 18 26 31 42 44; 10 12 19 25 32 41 45 54; 
          11 20 24 33 40 46 53 55; 21 23 34 39 47 52 56 61; 22 35 38 48 51 57 60 62; 36 37 49 50 58 59 63 64];
    for i = 1:size(zz,2)
        z = zz(:,i);
        coeffs(:,:,i) = reshape(z(ZZ(:)),8,8);
    end
end

function block = IDCT8x8(coeff)
    B=coeff;
    block(:,:,1) = idct2(B(:,:,1));
    block(:,:,2) = idct2(B(:,:,2));
    block(:,:,3) = idct2(B(:,:,3));
end

function quant = Quant8x8(dct_block, qScale)
    L = [16 11 10 16 24 40 51 61; 12 12 14 19 26 58 60 55; 14 13 16 24 40 57 69 56; 14 17 22 29 51 87 80 62;
         18 55 37 56 68 109 103 77; 24 35 55 64 81 104 113 92; 49 64 78 87 103 121 120 101; 72 92 95 98 112 100 103 99];
    
    C = [17 18 24 47 99 99 99 99; 18 21 26 66 99 99 99 99; 24 13 56 99 99 99 99 99; 47 66 99 99 99 99 99 99;
         99 99 99 99 99 99 99 99; 99 99 99 99 99 99 99 99; 99 99 99 99 99 99 99 99; 99 99 99 99 99 99 99 99];
    quant(:,:,1)= round(dct_block(:,:,1)./(L*qScale));
    quant(:,:,2)= round(dct_block(:,:,2)./(C*qScale));
    quant(:,:,3)= round(dct_block(:,:,3)./(C*qScale));
end

function zz = ZigZag8x8(quant)
    ZZ = [1 2 6 7 15 16 28 29; 3 5 8 14 17 27 30 43; 4 9 13 18 26 31 42 44; 10 12 19 25 32 41 45 54; 
          11 20 24 33 40 46 53 55; 21 23 34 39 47 52 56 61; 22 35 38 48 51 57 60 62; 36 37 49 50 58 59 63 64];
    for i = 1:size(quant,3)
        q = quant(:,:,i);
        z(ZZ(:)) = q(:);
        zz(:,i) = z;
    end
end

function zze = ZeroRunEnc_EoB(zz, EOB)
    block_nr = 1;
    zze=[];
    for block = 64:64:length(zz)
        zz_block = zz(block_nr:block);
        block_nr = block + 1;  
        zero_seq = 0;
        count = 0;
        for i = 1:length(zz_block)
            if zz_block(i) == 0
                if zero_seq == 1
                    count = count + 1;
                else
                    count = 0;
                    zero_seq = 1;
                end
                if i == length(zz_block)
                    zze(end+1) = EOB;
                end   
            else
                if zero_seq == 1
                    zze(end+1) = 0;
                    zze(end+1) = count;
                end
                   zero_seq = 0;
                   zze(end+1) = zz_block(i); 
            end
        end
    end
end

function dst = ZeroRunDec_EoB(src, EoB)
    count = 0;
    block = 1;
    dst=[];    
    while block <= length(src)
        if src(block) == 0
            dst(end+1:end+1+src(block+1)) = 0;
            count = count + src(block+1)+1;
            block = block + 2;
        elseif src(block) == EoB
            dst(end+1:end+(64-count)) = 0;
            block = block +1;
            count = 64;
        else
            dst(end+1) = src(block);
            block = block + 1;
            count = count + 1;
        end
        if count == 64
            count = 0;
        end
    end
end

function pmf = stats_marg(image, range)
    pmf = hist(image,range);
    pmf = pmf/sum(pmf);
end

function yuv = ictRGB2YCbCr(rgb)
    R = rgb(:,:,1);
    G = rgb(:,:,2);
    B = rgb(:,:,3);
    yuv(:,:,1) = 0.299*R+0.587*G+0.114*B;
    yuv(:,:,2) = -0.169*R-0.331*G+0.5*B;
    yuv(:,:,3) = 0.5*R-0.419*G-0.081*B;
end

function rgb = ictYCbCr2RGB(yuv)
    Y = yuv(:,:,1);
    Cb = yuv(:,:,2);
    Cr = yuv(:,:,3);
    rgb(:,:,1) = Y+1.402*Cr;
    rgb(:,:,2) = Y-0.344*Cb-0.714*Cr;
    rgb(:,:,3) = Y+1.772*Cb;
end