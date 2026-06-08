%% === ML Wavelet Shrinkage (Selected Signals, SNR = [3,5,7]) ===
clc; clear; close all force;

%% === Paths for HPRC ===
addpath('/scratch/user/vijinil/Dixon_Codes/WaveletSrinkage/LPM/matlab');
addpath('/scratch/user/vijinil/Dixon_Codes/WaveletSrinkage/NewCodes');
addpath('/scratch/user/vijinil/Dixon_Codes/WaveletSrinkage/Codes/codesMatLab5_2');
addpath('/scratch/user/vijinil/Dixon_Codes/wavelab850/Orthogonal');
addpath('/scratch/user/vijinil/Dixon_Codes/wavelab850/Utilities');
addpath('/scratch/user/vijinil/Dixon_Codes/wavelab850/DeNoising');
addpath('/scratch/user/vijinil/Dixon_Codes/WaveletSrinkage/Block_shrinkage');
addpath('/scratch/user/vijinil/Dixon_Codes');
addpath('/scratch/user/vijinil/viji');

%% === Output directory ===
outroot = '/scratch/user/vijinil/output_new/';
if ~exist(outroot, 'dir'), mkdir(outroot); end

%% === Parameters ===
SNRs = [3 5 7];
signals = {'Doppler','HeaviSine','Bumps','Blocks'};
methods = 1:5;
names = {'LR','SVM','RF','DT','NN'};
Cgrid = 0.2:0.2:2.0;
nsample = 100;
J = 10; n = 2^J; L = 5;

%% === Main loop across SNRs ===
for sIdx = 1:length(SNRs)
    snr = SNRs(sIdx);
    outdir = sprintf('%sSNR%d/', outroot, snr);
    if ~exist(outdir,'dir'), mkdir(outdir); end

    diary(fullfile(outdir, sprintf('runlog_SNR%d.txt', snr)));
    disp(['=== MATLAB JOB STARTED (SNR=' num2str(snr) ') ===']);

    bestC = zeros(numel(signals),1);
    bestM = strings(numel(signals),1);
    bestA = zeros(numel(signals),1);

    figBox = figure('Visible','off','Units','normalized','Position',[0.05 0.1 0.9 0.8],'Color','w');
    tiledlayout(2,2,'Padding','compact','TileSpacing','compact');

    for i = 1:numel(signals)
        fun = signals{i};
        fprintf('\n--- %s (SNR=%d) ---\n', fun, snr);

        %% Choose wavelet type
        switch fun
            case 'Blocks', wtype='Haar'; filtersize=2;
            case 'Bumps',  wtype='Daubechies'; filtersize=6;
            otherwise,     wtype='Symmlet'; filtersize=8;
        end

        filt = MakeONFilter(wtype, filtersize);
        yTrue0 = MakeSignal(fun, n);
        yTrue = sqrt(snr)/std(yTrue0) * yTrue0;

        %% === Search best classifier and c ===
        bestAMSE = inf; bestCval = NaN; bestMethod = NaN;
        for m = methods
            for ci = 1:length(Cgrid)
                cval = Cgrid(ci);
                lambda1 = sqrt(cval*log(n));
                lambda2 = sqrt(2*log(n));
                mseVals = zeros(1, nsample);

                for rep = 1:nsample
                    y = yTrue + randn(1,n);
                    [swt, model] = WaveletDenoise(y, L, filt, lambda1, lambda2, m);
                    yRec = idwtr(swt, L, filt);
                    mseVals(rep) = mean((yTrue - yRec).^2);
                end
                amse = mean(mseVals);
                if amse < bestAMSE
                    bestAMSE = amse; bestCval = cval; bestMethod = m;
                end
            end
        end

        bestC(i) = bestCval;
        bestM(i) = names{bestMethod};
        bestA(i) = bestAMSE;
        fprintf('%s → %s (c=%.2f, AMSE=%.4f)\n', fun, names{bestMethod}, bestCval, bestAMSE);

        %% === Boxplot (ML + 10 classical shrinkage rules) ===
        MSE = zeros(11, nsample);
        for rep = 1:nsample
            y = yTrue + randn(1,n);
            lambda1 = sqrt(bestCval*log(n)); lambda2 = sqrt(2*log(n));
            [swt, model] = WaveletDenoise(y, L, filt, lambda1, lambda2, bestMethod);
            yML = idwtr(swt, L, filt); MSE(1,rep)=mean((yTrue-yML).^2);

            yss = SemiSoftWaveletDenoise(y, L, filt, lambda1, lambda2);
            yss_rec = idwtr(yss, L, filt);
            MSE(2,rep)=mean((yTrue-yss_rec).^2);

            yhard = HardSoftThresholdings(y, L, filt, 1);     MSE(3,rep)=mean((yTrue-yhard).^2);

            L_bams = 7;
            ybams = BAMS(y, filt, L_bams);
            ybams = ybams(:)';
            MSE(4,rep)=mean((yTrue-ybams).^2);

            yCV = recdecompsh(y', filt);                      MSE(5,rep)=mean((yTrue-yCV).^2);

            thet0b =[0.5 1 0.7];
            ymed = recblockmed('Augment', y, [], thet0b);     MSE(6,rep)=mean((yTrue-ymed).^2);

            ymean = recblockmean('Augment', y, [], thet0b);   MSE(7,rep)=mean((yTrue-ymean).^2);

            theta0s = [0.5 1];
            yhyb = rechybblockmed('Augment', y, [], theta0s, thet0b);
            MSE(8,rep)=mean((yTrue-yhyb).^2);

            yblockJS = recblockJS('Augment', y, filt);        MSE(9,rep)=mean((yTrue-yblockJS).^2);

            yvisu = recvisu(y, 'H', filt);                    MSE(10,rep)=mean((yTrue-yvisu).^2);

            ygcv = recgcv(y, filt);                            MSE(11,rep)=mean((yTrue-ygcv).^2);
         end

        nexttile;
        boxplot(MSE','symbol','r+','Whisker',1.5,'Colors','b');
        title(sprintf('%s (c=%.2f, %s)', fun, bestCval, names{bestMethod}));
        set(gca,'FontSize',12);

        %% === Save progress after each signal ===
        save(fullfile(outdir,'SNR_progress.mat'), 'bestC','bestM','bestA');
    end

    %% === Save summary table ===
    fid = fopen(fullfile(outdir,'Summary.txt'),'w');
    fprintf(fid,'Signal\tBestC\tBestMethod\tAMSE\n');
    for i = 1:numel(signals)
        fprintf(fid,'%s\t%.2f\t%s\t%.4f\n',signals{i},bestC(i),bestM(i),bestA(i));
    end
    fclose(fid);
    save(fullfile(outdir, sprintf('Results_SNR%d.mat', snr)), 'bestC','bestM','bestA');

    sgtitle(sprintf('Boxplot Comparison (SNR=%d)', snr));
    saveas(figBox, sprintf('%sBoxplots_SNR%d.png', outdir, snr));
    close(figBox);

   %% === Reconstruction plots for SNR = 3, 5, 7 ===

snrList = [3 5 7];

for snr = snrList

    figRec = figure('Units','normalized','Position',[0.05 0.1 0.9 0.8],'Color','w');
    t = tiledlayout(2,2,'Padding','compact','TileSpacing','compact');

    for i = 1:numel(signals)

        bestCval = bestC(i);
        bestMethod = find(strcmp(names,bestM(i)));

        filt = MakeONFilter(wtype, filtersize);

        yTrue0 = MakeSignal(signals{i}, n);
        yTrue  = sqrt(snr)/std(yTrue0)*yTrue0;
        yNoisy = yTrue + randn(1,n);

        lambda1 = sqrt(bestCval*log(n));
        lambda2 = sqrt(2*log(n));

        [swt, model] = WaveletDenoise(yNoisy, L, filt, lambda1, lambda2, bestMethod);
        yRec = idwtr(swt, L, filt);

        nexttile;

        h1 = plot(yTrue,'k','LineWidth',1.5); hold on;
        h2 = plot(yRec,'r','LineWidth',1.2);
        h3 = scatter(1:n,yNoisy,5,'b','filled','MarkerFaceAlpha',0.4);

        xlim([1 n]);  % <<< FIX axis
        set(gca,'FontSize',12);
        title(signals{i});
    end

    % Shared legend
    lgd = legend([h1 h2 h3], {'Original','Reconstructed','Noisy'}, ...
        'Orientation','horizontal');
    lgd.Layout.Tile = 'north';

    sgtitle(sprintf('Original (black), Noisy (blue), Reconstructed (red) – SNR=%d', snr));

    saveas(figRec, sprintf('%sReconstruction_SNR%d.png', outdir, snr));
    close(figRec);

end
    disp(['=== MATLAB JOB FINISHED (SNR=' num2str(snr) ') ===']);
    diary off;
end
