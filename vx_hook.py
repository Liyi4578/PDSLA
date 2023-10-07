# -*- coding:utf-8 -*-
from unittest import runner
import requests
import time

from mmcv.parallel import is_module_wrapper
from mmcv.runner import HOOKS, Hook
from mmdet.core import DistEvalHook, EvalHook


@HOOKS.register_module()
class SetVXInfoHook(Hook):
	def __init__(self):
		self.eval_hook = None
		self.val_dataset = None
		
	def before_train_epoch(self, runner):
		epoch = runner.epoch
		model = runner.model
		if is_module_wrapper(model):
			model = model.module
		model.set_epoch(epoch)
		
	def before_train_iter(self, runner):
		cur_iter = runner.iter
		model = runner.model
		if is_module_wrapper(model):
			model = model.module
		model.set_iter(cur_iter)

	  
	def after_train_iter(self,runner):
		model = runner.model
		if is_module_wrapper(model):
			model = model.module
		fg_per_gt = model.get_fg_per_gt()
		if fg_per_gt is not None:
			num_fg_per_gt = fg_per_gt.get('num_fg_per_gt',0.0)
			num_fg_per_gt_aux = fg_per_gt.get('num_fg_per_gt_aux',None)
			runner.log_buffer.update({'num_fg_per_gt': float(num_fg_per_gt)},
											 runner.outputs['num_samples'])
			if num_fg_per_gt_aux is not None:
				runner.log_buffer.update({'num_fg_per_gt_aux': float(num_fg_per_gt_aux)},
										 runner.outputs['num_samples'])							  
	def _get_eval_results(self):
		"""Get model evaluation results."""
		results = self.eval_hook.latest_results
		eval_results = self.val_dataset.evaluate(
			results, logger='silent', **self.eval_hook.eval_kwargs)
		return eval_results
	
	
	def _send_info(self,runner,title:str,name:str,content:str):
		try:
			# Send the training info to my Weixin.
			ret = 0
			# ret = requests.post("xxx",
			#					 json={
			#						 "token": "xxx",
			#						 "title": title,
			#						 "name": name,
			#						 "content": content
			#					 })
			# runner.logger.info(ret.content.decode())
			# ret.raise_for_status()
		except requests.exceptions.ConnectionError:
			runner.logger.warning("ConnectionError: check your net")
			pass
		except requests.exceptions.HTTPError:
			runner.logger.warning("HTTPError: http return error code:{}".format(ret.status_code))
			pass
		except requests.TooManyRedirects:
			runner.logger.warning("TooManyRedirects!")
			pass
		except requests.ConnectTimeout:
			runner.logger.warning("ConnectTimeout!")
			pass
		except requests.ReadTimeout:
			runner.logger.warning("ReadTimeout!")
			pass
		except requests.RequestException:
			runner.logger.warning("RequestException!")
			pass
		except Exception:
			runner.logger.warning('VX Hook Error')
			pass
		
	def before_run(self,runner):
		if runner.rank != 0:
			return 
			
		for hook in runner.hooks:
			if isinstance(hook, (EvalHook, DistEvalHook)):
				self.eval_hook = hook
		if self.eval_hook is not None:
			self.val_dataset = self.eval_hook.dataloader.dataset
 
		exp_info = f'Exp name: {runner.meta["exp_name"]}'
		times = time.time()
		local_time = time.localtime(times)
		ttt = time.strftime("%Y-%m-%d %H:%M:%S",local_time)
		self._send_info(runner,title=exp_info,name="实验开始",content=f"start time:{ttt}")
			

	def after_train_epoch(self, runner):
		if runner.rank != 0:
			return 
		epoch = int(runner.epoch)

		log_dict = runner.log_buffer.output
		log_str = ''
		runner.logger.info(str(log_dict))
		for key in log_dict.keys():
			if key in ['bbox_mAP','bbox_mAP_50','bbox_mAP_75','bbox_mAP_s','bbox_mAP_m','bbox_mAP_l']:
				log_str += (key + ': '+ str(log_dict[key]) + ', ')
		exp_info = f'Exp name: {runner.meta["exp_name"]}'
		try:
			# Send the training info to my Weixin.
			ret = 0
			# ret = requests.post("https://www.autodl.com/api/v1/wechat/message/push",
			#					 json={
			#						 "token": "682d37e465da",
			#						 "title": exp_info,
			#						 "name": f"epoch-{epoch+1}",
			#						 "content": log_str
			#					 })
			# runner.logger.info(ret.content.decode())
			# ret.raise_for_status()
		except requests.exceptions.ConnectionError:
			runner.logger.warning("ConnectionError: check your net")
			pass
		except requests.exceptions.HTTPError:
			runner.logger.warning("HTTPError: http return error code:{}".format(ret.status_code))
			pass
		except requests.TooManyRedirects:
			runner.logger.warning("TooManyRedirects!")
			pass
		except requests.ConnectTimeout:
			runner.logger.warning("ConnectTimeout!")
			pass
		except requests.ReadTimeout:
			runner.logger.warning("ReadTimeout!")
			pass
		except requests.RequestException:
			runner.logger.warning("RequestException!")
			pass
		except Exception:
			runner.logger.warning('VX Hook Error')
			pass

		
if __name__ == '__main__':
	try:
		resp = 0
		# resp = requests.post("xxx",
							 # json={
								 # "token": "xxx",
								 # "title": 'test',
								 # "name": "hello",
								 # "content": 'hello world'
							 # })
		# print(resp.content.decode())
	except Exception:
		runner.logger.warning('VX Hook Error')
		pass