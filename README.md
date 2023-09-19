# ui2ai

This code creates desktop screen shots and trains the CNN with the screenshot data by having the user select a region on the screen shot that locates the relevant pixel array as coordinates for the Location of the Start Menu


update: so this is doing coordinate prediction, not image array analysis. Working this change to complete proof of concept. Will be looking into using Multi modal LLM's to attempt to workaround this need if it doesn't meet the criterea.


run "pip install -r requirements" to install the dependencies into a python environment of you choice.
This code has support for using CUDA to process tensor code.

Run win_ui.int.py to run the Gui app.
If you have dual monitors this app will default to open on secondary monitor as the code is setup to pull primary desktop images for training data.

The Model data is available here for the current training model:

Model.pth [https://drive.google.com/file/d/11HgElbdcJSEBGJbVeDfJ9AjM_Wj-rs-P/view?usp=sharing](https://1drv.ms/u/s!AiwPrNUOuUK-nLslRV_QISsaHeaGSA?e=LdhXvD)
model_context.json https://drive.google.com/file/d/104FHlYrnxPKOmC7t9yV1VkzqRkVqqG84/view?usp=sharing

Both files are required in the application root directory to use the "Find Start Menu" function.
