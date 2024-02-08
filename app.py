import PIL
import numpy as np
import gradio as gr

from algorithm.pipeline import Pipeline

class App:
    def __init__(self, 
                nmf='L1NormRegularizedNMF', 
                dataset='YaleB', 
                reduce=3, 
                noise_type='salt_and_pepper', 
                noise_level=0.10, 
                random_state=99, 
                scaler='MinMax'):
        self.pipeline = Pipeline(nmf=nmf, 
                            dataset=dataset, 
                            reduce=reduce, 
                            noise_type=noise_type, 
                            noise_level=noise_level, 
                            random_state=random_state, 
                            scaler=scaler)

    def align_reduce(self, dataset_name):
        return 1 if dataset_name == 'ORL' else 3

    def reset_pipeline(self, nmf, dataset, reduce, noise_type, noise_level, random_state, scaler):
        noise_type, noise_level = self.convert_level_to_number(noise_type, noise_level)
        self.pipeline = Pipeline(nmf=nmf, 
                            dataset=dataset, 
                            reduce=reduce, 
                            noise_type=noise_type, 
                            noise_level=noise_level, 
                            random_state=random_state, 
                            scaler=scaler)

    def convert_level_to_number(self, type, level):
        map_dict = {"Uniform": {"Low": 0.1, "High": 0.3}, 
                    "Gaussian": {"Low": 0.05, "High": 0.08}, 
                    "Laplacian": {"Low": 0.04, "High": 0.06}, 
                    "Salt & Pepper": {"Low": 0.02, "High": 0.1}, 
                    "Block": {"Low": 10, "High": 15}}
        type_name = type.lower() if type != "Salt & pepper" else "salt_and_pepper"
        return type_name, map_dict[type][level]

    def execute(self, max_iter=500, idx=9):
        self.pipeline.execute(max_iter=max_iter)
        return *self.visualize(idx), *self.metrics()

    def visualize(self, idx=9):
        image_raw, image_noise, image_recon = self.pipeline.visualization(idx=idx)
        return self.array2image(image_raw), self.array2image(image_noise), self.array2image(image_recon)

    def metrics(self):
        return self.pipeline.metrics
    
    def array2image(self, array):
        image_size = self.pipeline.img_size
        return PIL.Image.fromarray(self.scale_pixel(array)).resize((image_size))
    
    def scale_pixel(self, image):
        return ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
    
    def clear_params(self):
        self.pipeline = Pipeline()
        return 'L1NormRegularizedNMF', 'YaleB', 3, 'Salt & Pepper', 'Low', 99, 'MinMax'
    
app = App()
image_size = app.pipeline.img_size

with gr.Blocks() as demo:
    gr.Markdown("# NMF Image Reconstruction")
    with gr.Row():
        with gr.Group():
            with gr.Row():
                nmf = gr.Dropdown(
                    label="NMF Algorithm",
                    choices=['L1NormRegularizedNMF', 'L2NormNMF', 'KLDivergenceNMF', 
                            'ISDivergenceNMF', 'L21NormNMF', 'HSCostNMF', 
                            'CappedNormNMF', 'CauchyNMF'],
                    value='L1NormRegularizedNMF',
                    info="Choose the NMF algorithm.")
                
                dataset = gr.Dropdown(
                    label="Dataset",
                    choices=['ORL', 'YaleB'],
                    value='YaleB',
                    info="Choose the dataset.")
                
                reduce = gr.Number(
                    value=3,
                    label="Reduce",
                    info="Choose the reduce.")
                
            with gr.Row():
                noise_type = gr.Dropdown(
                    label="Noise Type",
                    choices=['Uniform', 'Gaussian', 'Laplacian', 'Salt & Pepper', 'Block'],
                    value='Salt & Pepper',
                    info="Choose the noise type.")
                
                noise_level = gr.Radio(
                    choices=['Low', 'High'],
                    value='Low',
                    label="Noise Level",
                    info="Choose the noise level."
                )

            with gr.Row():
                random_state = gr.Number(
                    value=99, 
                    label="Random State",
                    info="Choose the random state.",)
                
                scaler = gr.Dropdown(
                    label="Scaler",
                    choices=['MinMax', 'Standard'],
                    value='MinMax',
                    info="Choose the scaler.")
                
            with gr.Row():
                max_iter= gr.Number(
                    value=500,
                    label="Max Iteration",
                    info="Choose the max iteration.")
                idx = gr.Number(
                    value=9,
                    label="Image Index",
                    info="Choose the image index.")
            
            with gr.Row():
                execute_bt = gr.Button(value="Execute Algorithm",)
                clear_params_bt = gr.Button(
                    value="Clear Parameters")
        
        with gr.Group():
            with gr.Row():

                output_image_raw = gr.Image(
                    height=image_size[1],
                    width=image_size[0],
                    image_mode="L",
                    label="Original Image",
                    show_download_button=True,
                    show_share_button=True,)
                output_image_noise = gr.Image(
                    height=image_size[1],
                    width=image_size[0],
                    label="Noisy Image",
                    image_mode="L",
                    show_download_button=True,
                    show_share_button=True,)
                output_image_recon = gr.Image(
                    height=image_size[1],
                    width=image_size[0],
                    label="Reconstructed Image",
                    image_mode="L",
                    show_download_button=True,
                    show_share_button=True,)
                
            with gr.Row():
                rmse = gr.Number(
                    label="RMSE",
                    info="Average root mean square error",
                    precision=4,)
                acc = gr.Number(
                    label="Acc",
                    info="Accuracy",
                    precision=4,)
                nmi = gr.Number(
                    label="NMI",
                    info="Normalized mutual information",
                    precision=4,)

                clear_output_bt = gr.ClearButton(
                    value="Clear Output",
                components=[output_image_raw, output_image_noise, output_image_recon, rmse, acc, nmi],)
    
    nmf.input(app.reset_pipeline, inputs=[nmf, dataset, reduce, noise_type, noise_level, random_state, scaler])
    dataset.input(app.reset_pipeline, inputs=[nmf, dataset, reduce, noise_type, noise_level, random_state, scaler])
    dataset.input(app.align_reduce, inputs=[dataset], outputs=[reduce])
    reduce.input(app.reset_pipeline, inputs=[nmf, dataset, reduce, noise_type, noise_level, random_state, scaler])
    noise_type.input(app.reset_pipeline, inputs=[nmf, dataset, reduce, noise_type, noise_level, random_state, scaler])
    noise_level.input(app.reset_pipeline, inputs=[nmf, dataset, reduce, noise_type, noise_level, random_state, scaler])
    random_state.input(app.reset_pipeline, inputs=[nmf, dataset, reduce, noise_type, noise_level, random_state, scaler])
    scaler.input(app.reset_pipeline, inputs=[nmf, dataset, reduce, noise_type, noise_level, random_state, scaler])
    idx.input(app.visualize, inputs=[idx], outputs=[output_image_raw, output_image_noise, output_image_recon])
    execute_bt.click(app.execute, inputs=[max_iter, idx], outputs=[output_image_raw, output_image_noise, output_image_recon, rmse, acc, nmi])
    clear_params_bt.click(app.clear_params, outputs=[nmf, dataset, reduce, noise_type, noise_level, random_state, scaler])

if __name__ == '__main__':
    demo.queue()
    demo.launch(inbrowser=True,
                share=True,
                server_name="0.0.0.0",
                server_port=8080)