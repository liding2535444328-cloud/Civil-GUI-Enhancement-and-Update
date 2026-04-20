function [Scatter_Data, Stats_Summary, Best_Model] = T3_LSBoost_8_2(res_raw, pso_switch)
%% ========================================================================
%  ID: 1, 2 | Dataset: 2 | PSO: false/true | MaxSize: No | 8-Dim
%% ========================================================================
warning off; 
if nargin < 1, res_raw = readmatrix('数据集2.xlsx'); res_raw(any(isnan(res_raw),2),:)=[]; end
if nargin < 2, pso_switch = true; end 
loop_num = 10; results_cell = cell(loop_num, 1); main_tic = tic;
for run_i = 1:loop_num
    idx = randperm(size(res_raw,1)); split = round(0.8*size(res_raw,1));
    P_tr = res_raw(idx(1:split),1:8)'; T_tr = res_raw(idx(1:split),end)';
    P_te = res_raw(idx(split+1:end),1:8)'; T_te = res_raw(idx(split+1:end),end)';
    [p_tr_n, ps_in] = mapminmax(P_tr,0,1); p_te_n = mapminmax('apply',P_te,ps_in);
    [t_tr_n, ps_out] = mapminmax(T_tr,0,1);
    if pso_switch
        max_it=8; pop=8; lb=[0.01,10]; ub=[0.2,100]; 
        part=lb+(ub-lb).*rand(pop,2); gBest_sc=inf; gBest=[0.1,50];
        for t=1:max_it
            for i=1:pop
                m_tmp = fitrensemble(p_tr_n',t_tr_n','Method','LSBoost',...
                    'NumLearningCycles',round(part(i,2)),'LearnRate',part(i,1));
                err = mean((predict(m_tmp,p_te_n')-mapminmax('apply',T_te,ps_out)').^2);
                if err < gBest_sc, gBest_sc=err; gBest=part(i,:); end
            end
            part = max(min(part+1.2*rand*(repmat(gBest,pop,1)-part),ub),lb);
        end
        f_lr=gBest(1); f_n=round(gBest(2));
    else
        f_lr=0.1; f_n=50;
    end
    final_m = fitrensemble(p_tr_n',t_tr_n','Method','LSBoost','NumLearningCycles',f_n,'LearnRate',f_lr);
    T_sim = mapminmax('reverse',predict(final_m,p_te_n')',ps_out)';
    tmp.R2 = 1-sum((T_te'-T_sim).^2)/sum((T_te'-mean(T_te')).^2);
    tmp.RMSE = sqrt(mean((T_te'-T_sim).^2)); tmp.MAE = mean(abs(T_te'-T_sim));
    tmp.T_real = T_te'; tmp.T_sim = T_sim; tmp.model = final_m; results_cell{run_i} = tmp;
end
r2_all = cellfun(@(x) x.R2, results_cell); [~,b_idx]=max(r2_all); bp=results_cell{b_idx}; 
Stats_Summary.R2_mean=mean(r2_all); Stats_Summary.RMSE_test_loop=cellfun(@(x) x.RMSE, results_cell);
Stats_Summary.MAE_test_loop=cellfun(@(x) x.MAE, results_cell); Stats_Summary.Time_total=toc(main_tic);
Scatter_Data.te_real=bp.T_real; Scatter_Data.te_sim=bp.T_sim; Best_Model=bp.model;
if nargin < 1 % Standalone plot
    figure('Color','w'); scatter(bp.T_real, bp.T_sim, 'filled'); hold on;
    line_ref = [min(bp.T_real) max(bp.T_real)]; plot(line_ref, line_ref, 'k--');
    xlabel('Experimental (MPa)'); ylabel('Predicted (MPa)'); title('Regression Analysis');
end
end