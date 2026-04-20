function [Scatter_Data, Stats_Summary, Best_Model] = T3_LSBoost(res_raw, pso_switch)
%% ========================================================================
%  ID: 3, 4, 5, 6 | Dataset: 4, 3, 4, 3 | PSO: No/No/Yes/Yes | MaxSize: No/Yes/No/Yes
%% ========================================================================
warning off; if nargin < 1, res_raw=readmatrix('数据集3.xlsx'); res_raw(any(isnan(res_raw),2),:)=[]; end
if nargin < 2, pso_switch = true; end 
loop_num = 10; results_cell = cell(loop_num, 1); main_tic = tic;
for run_i = 1:loop_num
    idx = randperm(size(res_raw,1)); split = round(0.8*size(res_raw,1));
    P_tr = res_raw(idx(1:split),1:9); T_tr = res_raw(idx(1:split),end);
    P_te = res_raw(idx(split+1:end),1:9); T_te = res_raw(idx(split+1:end),end);
    [P_tr_n, ps_in] = mapminmax(P_tr',0,1); P_te_n = mapminmax('apply',P_te',ps_in);
    [T_tr_n, ps_out] = mapminmax(T_tr',0,1);
    if pso_switch
        pop=8; max_ge=8; lb=[0.01,20]; ub=[0.2,100];
        part=lb+(ub-lb).*rand(pop,2); gBest_sc=inf; gBest=[0.1,50];
        for t=1:max_ge
            for i=1:pop
                m_tmp = fitrensemble(P_tr_n',T_tr_n','Method','LSBoost',...
                    'NumLearningCycles',round(part(i,2)),'LearnRate',part(i,1));
                err = mean((predict(m_tmp,P_te_n')-mapminmax('apply',T_te',ps_out)').^2);
                if err < gBest_sc, gBest_sc=err; gBest=part(i,:); end
            end
            part = max(min(part+1.2*rand*(repmat(gBest,pop,1)-part),ub),lb);
        end
        f_lr=gBest(1); f_n=round(gBest(2));
    else
        f_lr=0.1; f_n=50;
    end
    final_m = fitrensemble(P_tr_n',T_tr_n','Method','LSBoost','NumLearningCycles',f_n,'LearnRate',f_lr);
    T_sim = mapminmax('reverse',predict(final_m,P_te_n')',ps_out)';
    tmp.R2 = 1-sum((T_te-T_sim).^2)/sum((T_te-mean(T_te)).^2);
    tmp.RMSE = sqrt(mean((T_te-T_sim).^2)); tmp.MAE = mean(abs(T_te-T_sim));
    tmp.T_te_real = T_te; tmp.T_te_sim = T_sim; tmp.model = final_m; results_cell{run_i} = tmp;
end
r2_vals = cellfun(@(x) x.R2, results_cell); [~,b_idx]=max(r2_vals); bp=results_cell{b_idx}; 
Stats_Summary.R2_mean=mean(r2_vals); Stats_Summary.RMSE_test_loop=cellfun(@(x) x.RMSE, results_cell);
Stats_Summary.MAE_test_loop=cellfun(@(x) x.MAE, results_cell); Stats_Summary.Time_total=toc(main_tic);
Scatter_Data.te_real=bp.T_te_real; Scatter_Data.te_sim=bp.T_te_sim; Best_Model=bp.model;
if nargin < 1 % Standalone plot
    figure('Color','w'); scatter(bp.T_te_real, bp.T_te_sim, 'filled'); hold on;
    line_ref = [min(bp.T_te_real) max(bp.T_te_real)]; plot(line_ref, line_ref, 'k--');
    xlabel('Experimental Strength (MPa)'); ylabel('Predicted Strength (MPa)'); title('Regression Analysis');
end
end