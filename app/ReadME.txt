stags:
- backend:
	- flask server - main_server.py
	- Needs https support ofcourse - and new model
	- just running the file with python3
	- using the model staticly, need to change by hand for different one.
- frontend:
	- chat-ui
	- React app sending the questions to the backend
	- Running - Opening powershell in the folder and running "npm start"
	- http://localhost:3000/ - On edge only, chrome blocked my local running because of bug

Thoughts:
	- Dataset needs to have roles
		- The add roles script faild in one point, need to be run again
	- Dataset needs to be Hebrew and English
		- Change the translate script to add the Hebrew and not replace the dataset
	- Rephrasing
		- Need to be run again before the translations and after roles
		- Failed for length, fixed and added a bit of logging
	- Reran the model from scratch on the new one, and the one more for fine-tuning
	- Need to improve speech to text feature, it works but maybe could be better
	- understand how to export the server outside
	- understand how to run the server with encryption from one end to the other

Idea:
	- Take the model https://huggingface.co/vdpappu/lora_psychology-2
		- Add a good translator, add before everything to English And turn it back to the source
			- We can a labler model for choosing language haha