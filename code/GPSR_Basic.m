function [x,x_debias,objective,times,debias_start,mses]= ...
    GPSR_Basic(y,A,tau,varargin)

stopCriterion = 3;
tolA = 0.01;
debias = 0;
maxiter = 10000;
miniter = 5;
init = 0;
compute_mse = 0;
AT = 0;
verbose = 1;
continuation = 0;
cont_steps = 5;
firstTauFactorGiven = 0;
Initial_X_supplied = 3;

mu = 0.1;
lambda_backtrack = 0.5;

debias_start = 0;
x_debias = [];
mses = [];

if (rem(length(varargin),2)==1)
  error('Optional parameters should always go by pairs');
else
  for i=1:2:(length(varargin)-1)
    switch upper(varargin{i})
     case 'STOPCRITERION'
       stopCriterion = varargin{i+1};
     case 'TOLERANCEA'       
       tolA = varargin{i+1};
     case 'DEBIAS'
       debias = varargin{i+1};
     case 'MAXITERA'
       maxiter = varargin{i+1};
     case 'MINITERA'
       miniter = varargin{i+1};
     case 'INITIALIZATION'
       if prod(size(varargin{i+1})) > 1
          init = Initial_X_supplied;
          x = varargin{i+1};
       else 
          init = varargin{i+1};
       end
     case 'CONTINUATION'
       continuation = varargin{i+1};  
     case 'CONTINUATIONSTEPS' 
       cont_steps = varargin{i+1};
     case 'FIRSTTAUFACTOR'
       firstTauFactor = varargin{i+1};
       firstTauFactorGiven = 1;
     case 'TRUE_X'
       compute_mse = 1;
       true = varargin{i+1};
     case 'AT'
       AT = varargin{i+1};
     case 'VERBOSE'
       verbose = varargin{i+1};
     otherwise
      % Hmmm, something wrong with the parameter string
      error(['Unrecognized option: ''' varargin{i} '''']);
    end
  end
end
%%%%%%%%%%%%%%

if ~isa(A, 'function_handle')
   AT = @(x) A'*x;
   A = @(x) A*x;
end

Aty = AT(y);

switch init
    case 0   % initialize at zero
       x = AT(zeros(size(y)));
    case 1   % initialize randomly
       x = randn(size(AT(zeros(size(y)))));
    case 2   % initialize x0 = A'*y
       x = Aty;
   case Initial_X_supplied
       % initial x was given as a function argument
       if size(A(x)) ~= size(y)
          error(['Size of initial x is not compatible with A']); 
       end
end
      

if prod(size(tau)) == 1
   aux = AT(y);
   max_tau = max(abs(aux(:)));
   if tau >= max_tau
      x = zeros(size(aux));
      if debias
         x_debias = x;
      end
      objective(1) = 0.5*(y(:)'*y(:));
      times(1) = 0;
      if compute_mse
          mses(1) = sum(true(:).^2);
      end
      return
   end 
end

% initialize u and v
u =  x.*(x >= 0);
v = -x.*(x <  0);

nz_x = (x ~= 0.0);
num_nz_x = sum(nz_x(:));

resid =  y - A(x);
f = 0.5*(resid(:)'*resid(:)) + sum(tau(:).*u(:)) + sum(tau(:).*v(:));

t0 = cputime;

final_tau = tau;

final_stopCriterion = stopCriterion;
final_tolA = tolA;

if continuation&(cont_steps > 1)
   if prod(size(tau)) == 1
      if (firstTauFactorGiven == 0)|(firstTauFactor*tau >= max_tau)
         firstTauFactor = 0.8*max_tau / tau;
         fprintf(1,'parameter FirstTauFactor too large; changing')
      end
   end
   cont_factors = 10.^[log10(firstTauFactor):...
                    log10(1/firstTauFactor)/(cont_steps-1):0];
else
  cont_factors = 1;
  cont_steps = 1;
end
  

iter = 1;
if compute_mse
   mses(iter) = sum((x(:)-true(:)).^2);
end

for cont_loop = 1:cont_steps

    tau = final_tau * cont_factors(cont_loop);
    
    if verbose
        fprintf(1,'\nSetting tau = %8.4f\n',tau)
    end
    
    if cont_loop == cont_steps
       stopCriterion = final_stopCriterion;
       tolA = final_tolA;
    else 
       stopCriterion = 3;
       tolA = 1e-3;
    end
    
    resid =  y - A(x);
    f = 0.5*(resid(:)'*resid(:)) + ...
             sum(tau(:).*u(:)) + sum(tau(:).*v(:));

    objective(iter) = f;
    times(iter) = cputime - t0;
    
    resid_base = y - resid;

    keep_going = 1;

    if verbose
       fprintf(1,'\nInitial obj=%10.6e, nonzeros=%7d\n', f,num_nz_x);
    end

    while keep_going

      x_previous = x;

      temp = AT(resid_base); %ATA(u - v)
      term  =  temp - Aty;
      gradu =  term + tau;
      gradv = -term + tau;

      dx = gradv-gradu;
      old_u = u; old_v = v;

      auv = A(dx);
      dGd = auv(:)'*auv(:);

      condgradu = ((old_u>0) | (gradu<0)) .* gradu;
      condgradv = ((old_v>0) | (gradv<0)) .* gradv;
      auv_cond = A(condgradu-condgradv);
      dGd_cond = auv_cond(:)'*auv_cond(:);
      lambda0 = (gradu(:)'*condgradu(:) + gradv(:)'*condgradv(:)) /...
                (dGd_cond + realmin);

      lambda = lambda0; 
      while 1
        du = max(u-lambda*gradu,0.0) - u; 
        u_new = u + du;
        dv = max(v-lambda*gradv,0.0) - v; 
        v_new = v + dv;
        dx = du-dv; 
        x_new = x + dx;

        resid_base = A(x_new);
        resid = y - resid_base;
        f_new = 0.5*(resid(:)'*resid(:)) +  ...
        sum(tau(:).*u_new(:)) + sum(tau(:).*v_new(:));    
        if f_new <= f + mu * (gradu'*du + gradv'*dv)
          %disp('OK')  
          break
        end
        lambda = lambda * lambda_backtrack;
        fprintf(1,'    reducing lambda to %6.2e\n', lambda)
      end
      u = u_new; 
      v = v_new; 
      prev_f = f; 
      f = f_new;
      uvmin = min(u,v); 
      u = u - uvmin; 
      v = v - uvmin; 
      x = u-v;

      nz_x_prev = nz_x;
      nz_x = (x~=0.0);
      num_nz_x = sum(nz_x(:));
      
      iter = iter + 1;
      objective(iter) = f;
      times(iter) = cputime-t0;
      lambdas(iter) = lambda;

      if compute_mse
        err = true - x;
        mses(iter) = (err(:)'*err(:));
      end
      
      if verbose
	fprintf(1,'It =%4d, obj=%9.5e, lambda=%6.2e, nz=%8d   ',...
	    iter, f, lambda, num_nz_x);
      end
      
    switch stopCriterion
	case 0
      num_changes_active = (sum(nz_x(:)~=nz_x_prev(:)));
      if num_nz_x >= 1
         criterionActiveSet = num_changes_active;
      else
         criterionActiveSet = tolA / 2;
      end
      keep_going = (criterionActiveSet > tolA);
      if verbose
         fprintf(1,'Delta n-zeros = %d (target = %e)\n',...
                criterionActiveSet , tolA) 
      end
	case 1
	  criterionObjective = abs(f-prev_f)/(prev_f);
      keep_going =  (criterionObjective > tolA);
      if verbose
         fprintf(1,'Delta obj. = %e (target = %e)\n',...
                criterionObjective , tolA) 
      end
	case 2
      delta_x_criterion = norm(dx(:))/norm(x(:));
      keep_going = (delta_x_criterion > tolA);
      if verbose
         fprintf(1,'Norm(delta x)/norm(x) = %e (target = %e)\n',...
             delta_x_criterion,tolA) 
      end
	case 3
        w = [ min(gradu(:), old_u(:)); min(gradv(:), old_v(:)) ];
        criterionLCP = norm(w(:), inf);
        criterionLCP = criterionLCP / ...
                 max([1.0e-6, norm(old_u(:),inf), norm(old_v(:),inf)]);
       keep_going = (criterionLCP > tolA);
       if verbose
         fprintf(1,'LCP = %e (target = %e)\n',criterionLCP,tolA) 
       end
    case 4
      keep_going = (f > tolA);
      if verbose
         fprintf(1,'Objective = %e (target = %e)\n',f,tolA) 
      end
    end 
    
    if iter<=miniter
       keep_going = 1;
    else
        if iter > maxiter
            keep_going = 0;
        end
    end

    end

end

if verbose
  fprintf(1,'\nFinished the main algorithm!\nResults:\n')
  fprintf(1,'||A x - y ||_2^2 = %10.3e\n',resid(:)'*resid(:))
  fprintf(1,'||x||_1 = %10.3e\n',sum(abs(x(:))))
  fprintf(1,'Objective function = %10.3e\n',f);
  nz_x = (x~=0.0); num_nz_x = sum(nz_x(:));
  fprintf(1,'Number of non-zero components = %d\n',num_nz_x);
  fprintf(1,'CPU time so far = %10.3e\n', times(iter));
  fprintf(1,'\n');
end

end
