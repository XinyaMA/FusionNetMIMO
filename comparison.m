clear all
clc

% Parameters
Nt = 2;
Nr = 2;
M = 16;
k = log2(M);
snr_dB = 0:2:20;
numSym = 1000;

berML = zeros(size(snr_dB));
berZF = zeros(size(snr_dB));
berMMSE = zeros(size(snr_dB));
berSIC = zeros(size(snr_dB));
berSDR = zeros(size(snr_dB));

% Start of simulation
for s = 1:length(snr_dB)
    snrLin = 10^(snr_dB(s)/10);
    noise_var = 1/snrLin;

    bitErrML = 0;
    bitErrZF = 0;
    bitErrMMSE = 0;
    bitErrSIC = 0;
    totalBits = numSym * Nt * k;

    for i = 1:numSym
        % Generating Symbols
        txSym = randi([0 M-1], Nt, 1);
        txSig = qammod(txSym, M, 'UnitAveragePower', true);
        txBits = reshape(de2bi(txSym, k, 'left-msb').', [], 1);

        % Channels and Noise
        H = (randn(Nr,Nt) + 1j*randn(Nr,Nt)) / sqrt(2);
        n = sqrt(noise_var/2)*(randn(Nr,1) + 1j*randn(Nr,1));
        y = H * txSig + n;

        % ZF
        zf_est = pinv(H) * y;
        zf_sym = qamdemod(zf_est, M, 'UnitAveragePower', true);
        zfBits = reshape(de2bi(zf_sym, k, 'left-msb').', [], 1);
        bitErrZF = bitErrZF + sum(zfBits ~= txBits);

        % MMSE
        mmse_est = (H'*H + noise_var*eye(Nt)) \ H' * y;
        mmse_sym = qamdemod(mmse_est, M, 'UnitAveragePower', true);
        mmseBits = reshape(de2bi(mmse_sym, k, 'left-msb').', [], 1);
        bitErrMMSE = bitErrMMSE + sum(mmseBits ~= txBits);

        % ML
        minDist = inf;
        bestSym = zeros(Nt,1);
        for s1 = 0:M-1
            for s2 = 0:M-1
                testSym = [s1; s2];
                testSig = qammod(testSym, M, 'UnitAveragePower', true);
                dist = norm(y - H*testSig)^2;
                if dist < minDist
                    minDist = dist;
                    bestSym = testSym;
                end
            end
        end
        mlBits = reshape(de2bi(bestSym, k, 'left-msb').', [], 1);
        bitErrML = bitErrML + sum(mlBits ~= txBits);

        % SIC
        [~, order] = sort(vecnorm(H).^2, 'descend');
        H_sic = H;
        y_sic = y;
        detected = zeros(Nt,1);

        for j = 1:Nt
            idx = order(j);
            h_j = H_sic(:,idx);
            s_est = (h_j'*h_j)\(h_j'*y_sic);
            detected(idx) = qamdemod(s_est, M, 'UnitAveragePower', true);
            s_mod = qammod(detected(idx), M, 'UnitAveragePower', true);
            y_sic = y_sic - h_j * s_mod;
            H_sic(:,idx) = zeros(Nr,1);
        end
        sicBits = reshape(de2bi(detected, k, 'left-msb').', [], 1);
        bitErrSIC = bitErrSIC + sum(sicBits ~= txBits);
    end

    berML(s) = bitErrML / totalBits;
    berZF(s) = bitErrZF / totalBits;
    berMMSE(s) = bitErrMMSE / totalBits;
    berSIC(s) = bitErrSIC / totalBits;

    fprintf("SNR=%2d dB | ML: %.4e, ZF: %.4e, MMSE: %.4e, SIC: %.4e\n", ...
        snr_dB(s), berML(s), berZF(s), berMMSE(s), berSIC(s));
end

% SDR main loop
for s = 1:length(snr_dB)
    snrLin = 10^(snr_dB(s)/10);
    noise_var = 1/snrLin;

    bitErr = 0;
    totalBits =  numSym * Nt * k;

    for i = 1:numSym
        txSym = randi([0 M-1], Nt, 1);
        txSig = qammod(txSym, M, 'UnitAveragePower', true);
        txBits = reshape(de2bi(txSym, k, 'left-msb').', [], 1);

        H = (randn(Nr,Nt) + 1j*randn(Nr,Nt)) / sqrt(2);
        noise = sqrt(noise_var/2)*(randn(Nr,1) + 1j*randn(Nr,1));
        y = H * txSig + noise;

        % SDR detection
        sdrSym = sdr_detector(H, y, M);
        sdrBits = reshape(de2bi(sdrSym, k, 'left-msb').', [], 1);

        bitErr = bitErr + sum(sdrBits ~= txBits);
    end

    berSDR(s) = bitErr / totalBits;
    fprintf('SNR = %2d dB | SDR BER = %.4e\n', snr_dB(s), berSDR(s));
end

% Comparison
figure;
semilogy(snr_dB, berML, '-o', 'LineWidth', 1.5); hold on;
semilogy(snr_dB, berZF, '-s', 'LineWidth', 1.5);
semilogy(snr_dB, berMMSE, '-^', 'LineWidth', 1.5);
semilogy(snr_dB, berSIC, '-d', 'LineWidth', 1.5);
semilogy(snr_dB, berSDR, '-x', 'LineWidth', 1.5);
grid on;
legend('ML', 'ZF', 'MMSE', 'SIC','SDR');
xlabel('SNR (dB)');
ylabel('Bit Error Rate (BER)');
title('BER Comparison of MIMO Detection Algorithms');

function detected_sym = sdr_detector(H, y, M)
    Nt = size(H,2); % Number of transmitting antennas
    constellation = qammod(0:M-1, M, 'UnitAveragePower', true);
    k = log2(M);

    % Converting complex number problems to real number form
    H_r = [real(H), -imag(H); imag(H), real(H)];
    y_r = [real(y); imag(y)];
    A = H_r' * H_r;
    b = -2 * H_r' * y_r;
    c = y_r' * y_r;
    n = 2 * Nt;

    % Solving SDP using CVX
    cvx_begin sdp quiet
        variable X(n+1, n+1) symmetric
        minimize( trace([A b/2; b'/2 c] * X) )
        subject to
            X(n+1,n+1) == 1;
            X >= 0;
    cvx_end

    % approximate solution
    x_hat = X(1:n,n+1);
    x_hat_cplx = x_hat(1:Nt) + 1i * x_hat(Nt+1:end);

    % Hard verdicts mapped to constellation symbols
    detected_sym = zeros(Nt,1);
    for i = 1:Nt
        [~, idx] = min(abs(constellation - x_hat_cplx(i)));
        detected_sym(i) = idx - 1;
    end
end
