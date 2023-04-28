from model import *
from torch.distributions import Categorical
from torch.distributions.kl import kl_divergence

class PPO(nn.Module):
    def __init__(self,**config):
        super().__init__()
        self.actor                  = ActorModel()
        self.critic                 = CriticModel()

        self.num_games              = config['num_games']
        self.n_game_per_batch       = config['num_game_per_batch']
        self.num_epochs             = config['num_epochs']
        self.lr                     = config['lr']
        self.batch_size             = config['batch_size']
        self.entropy_coef           = config['entropy_coef']
        self.critic_coef            = config['critic_coef']
        self.gamma                  = config['gamma']
        self.gae_lambda             = config['gae_lambda']
        self.PPO_params             = list(self.actor.parameters()) + list(self.critic.parameters())
        self.optimizer              = torch.optim.Adam(self.PPO_params,lr=self.lr)

        self.policy_kl_range        = config['policy_kl_range']
        self.policy_params          = config['policy_params']

        self.batch_states           = []
        self.batch_actions          = []
        self.batch_log_probs        = []
        self.batch_values           = []
        self.batch_mask             = []
        self.batch_rtgs             = []

    def GetPolicy(self,state):
        return self.actor.Forward(state)
    
    def GetValue(self,state):
        return self.critic.Forward(state)
    
    def ClearBatch(self):
        self.batch_states           = []
        self.batch_actions          = []
        self.batch_log_probs        = []
        self.batch_values           = []
        self.batch_mask             = []
        self.batch_rtgs             = []

    def ToTensor(self):
        self.batch_states           = torch.tensor(self.batch_states,dtype=torch.float32).detach()
        self.batch_actions          = torch.tensor(self.batch_actions,dtype=torch.float32).detach()
        self.batch_log_probs        = torch.tensor(self.batch_log_probs,dtype=torch.float32).detach()
        self.batch_values           = torch.tensor(self.batch_values,dtype=torch.float32).detach()
        self.batch_mask             = torch.tensor(self.batch_mask,dtype=torch.float32).detach()
        self.batch_rtgs             = torch.tensor(self.batch_rtgs,dtype=torch.float32).detach()
    
    def GAE(self,rewards,values,is_terminal,gamma,gae_lambda):
        """General Advantage Estimation"""
        advantages      = np.zeros_like(rewards)
        last_advantage  = 0
        last_value      = values[-1]

        for t in range(len(rewards)-1,-1,-1):

            mask            = 1.0 - is_terminal[t]
            last_value      = last_value * mask
            last_advantage  = last_advantage * mask
            delta           = rewards[t] + gamma * last_value - values[t]
            last_advantage  = delta + gamma * gae_lambda * last_advantage
            advantages[t]   = last_advantage
            last_value      = values[t]
        
        return advantages
    
    def MonteCarloRewards(self,rewards,is_terminals,gamma):
        """Reward-to-go"""
        rtgs = []
        discounted_reward = 0
        # rewards = (rewards - rewards.mean()) / (rewards.std()+1e-10)
        for reward, is_terminal in np.column_stack((rewards[::-1],is_terminals[::-1])):
            if is_terminal==1:
                discounted_reward = 0
            discounted_reward = reward + (gamma * discounted_reward)
            rtgs.append(discounted_reward)
        return rtgs
    
    def BatchLoader(self,states,actions,logprobs,rtgs,mask,value,batch_size=64):
        """Batch Loader"""
        state,action,prob = states,actions,logprobs
        n_samples         = state.shape[0]
        # print(n_samples)
        index = torch.randperm(n_samples)

        state,action,prob,rtgs,mask,value= state[index],action[index],prob[index],rtgs[index],mask[index],value[index]
        for i in np.arange(0, n_samples, batch_size):
            begin, end = i, min(i + batch_size, n_samples)
            yield state[begin:end],action[begin:end],prob[begin:end],rtgs[begin:end],mask[begin:end],value[begin:end]


    
    def Normalized(self,x):
        return (x - x.mean()) / (x.std() + 1e-8)
    
    def CalculateTrulyLoss(self,value,value_new,entropy,log_prob,log_prob_new,rtgs):
        """Calculate Model Loss"""
        advantage       = rtgs - value.detach()
        ratios          = torch.exp(torch.clamp(log_prob_new-log_prob.detach(),min=-20.,max=5.))
        Kl              = kl_divergence(Categorical(logits=log_prob), Categorical(logits=log_prob_new))

        actor_loss      = -torch.where(
                            (Kl >= self.policy_kl_range) & (ratios >= 1),
                            ratios * advantage - self.policy_params * Kl,
                            ratios * advantage
                        ).mean()
        value_clipped   = value + torch.clamp(value_new - value, -self.value_clip, self.value_clip)

        critic_loss     = 0.5 * torch.max((rtgs-value_new)**2,(rtgs-value_clipped)**2).mean()
        total_loss      = actor_loss + self.critic_coef * critic_loss - self.entropy_coef * entropy
        return total_loss

    def TrainModel(self):
        """Training Model"""
        self.ToTensor()
        for i in range(self.num_epochs):
            for state,action,log_prob,rtgs,mask,value in self.BatchLoader(self.batch_states,self.batch_actions,self.batch_logprobs,self.batch_rtgs,self.batch_mask,self.batch_values,batch_size=self.batch_size):
                
                policy,value_new  = self.Forward(state)
                value_new         = value_new.squeeze(1)
                value             = value.squeeze(1)
                prob1             = Categorical(logits=policy+ torch.log(mask))
                log_prob_new      = prob1.log_prob(action.view(1,-1)).squeeze()
                entropy           = prob1.entropy()
                total_loss        = self.CalculateTrulyLoss(value,value_new,entropy,log_prob,log_prob_new,rtgs)
                
                if not torch.isnan(total_loss).any():
                    self.optimizer.zero_grad(set_to_none=True)
                    total_loss.mean().backward()
                    nn.utils.clip_grad_norm_(self.parameters(),0.5)
                    self.optimizer.step()

    
    

    