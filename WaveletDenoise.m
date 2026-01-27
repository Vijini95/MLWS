function swt = WaveletDenoise(yNoisy, L, filt,lambda1,lambda2, isMethod)

                wtdata = dwtr(yNoisy, L, filt);
                n = length(wtdata);
        
                finest = wtdata(n/2+1:n);
                sigmahat = std(finest);

                %sigmahat = mad(finest)/0.6757;
        
                lambda_1n = lambda1 * sigmahat; 
                lambda_2n = lambda2 * sigmahat;
                
                swt3 = wtdata .* (abs(wtdata) <= lambda_2n & abs(wtdata) >= lambda_1n );
                swt2 = wtdata .* (abs(wtdata) > lambda_2n );
                swt1 = wtdata .* (abs(wtdata) < lambda_1n );
                
                id_3 = find(swt3); id_2 = find(swt2); id_1 = find(swt1);
                
                X = [];
        
                for k = 1 : n
        
                    if ismember(k, id_1)
                        if k == 1 || k == n
                            x1 = [ wtdata(k)  wtdata(k)/2 L 0];
                        else
                            x1 = [ wtdata(k) (wtdata(k-1) + wtdata(k+1))/2 L 0];
                        end
                        X = [X;x1];
        
                    elseif ismember(k, id_2)
                        if k == 1 || k == n
                            x1 = [ wtdata(k)  wtdata(k)/2 L 1];
                        else
                            x1 = [ wtdata(k) (wtdata(k-1) + wtdata(k+1))/2 L 1];
                        end
                        X = [X;x1];
        
                    elseif ismember(k, id_3)
                        if k == 1 || k == n
                            x1 = [ wtdata(k)  wtdata(k)/2 L 2];
                        else
                            x1 = [ wtdata(k) (wtdata(k-1) + wtdata(k+1))/2 L 2];
                        end
                        X = [X;x1];
                    end
                    
                end
                Xtrain = X(find(X(:,end) ~= 2 ),:); 
                Xtest = X(find(X(:,end) == 2 ),1:end-1);
                
                if isMethod == 1
                    id3_pred_class = LogisticRegModel(Xtrain, Xtest, 0);
                elseif isMethod  == 2
                    [id3_pred_class, model, performance_metrics] = SVMModel(Xtrain, Xtest, 0);
                elseif isMethod == 3
                    [id3_pred_class, model] =  RFModel(Xtrain, Xtest,0);
                end 
                
                sw3_updated = swt3(id_3).*id3_pred_class'; 
                
                swt = zeros(1, n);
                %swt(id_1) = wtdata(id_1); 
                swt(id_2) = wtdata(id_2);  
                swt( id_3(find(sw3_updated)) ) = wtdata(id_3(find(sw3_updated)));
end
