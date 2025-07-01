<h1 align="center">
     Rethink Sparse Signals for Pose-guided Text-to-image Generation
</h1></h1> 
<p align="center">
<a href="https://arxiv.org/abs/2506.20983"><img src="https://img.shields.io/badge/arXiv-2506.20983-b31b1b.svg"></a>
</p>


> Wenjie Xuan, Jing Zhang, Juhua Liu, Bo Du, Dacheng Tao

This is the official implementation for the paper "*Rethink Sparse Signals for Pose-guided Text-to-image Generation*", accepted by ICCV 2025. We propose a Spatial-Pose ControlNet, namely *SP-Ctrl*, which equips sparse signals with robust controllability for pose-guided image generation. Our work features an improved spatial-pose representation and a keypoint-concept learning and alignment strategy. Please refer to our paper for more details.



## :fire: News

- **[2025/06/26]**: Our work is accepted by ICCV 2025. We are preparing to release the code in a few weeks. 



## :round_pushpin: Todo

- [ ] Release the code for preparing datasets. 
- [ ] Release the code for training. 
- [ ] Release the code and checkpoints for inference. 



## :sparkles: Highlight

![figure_1](assets\figure_1.png)

- **We rethink the sparse signals for pose-guided T2I generation and propose *SP-Ctrl*,** a spatial-pose ControlNet that enables precise pose alignment with sparse signals. It reveals the potential of sparse signals in spatially controllable generations. 
- **A Spatial-Pose Representation for better pose instructions**. We introduce a Spatial-Pose Representation with learnable keypoint embeddings to enhance the expressiveness of sparse pose signals.
- **A Keypoint-Concept Learning strategy to encourage text-spatial alignment**. We propose Keypoint Concept Learning, a novel strategy that enhances keypoint focus and discrimination, enhancing details and improving pose alignment.
- **Better pose accuracy, diversity and cross-species generation ability**. Experiments on animal- and human-centric T2I generation validate that our method achieves performance comparable to dense signals. Moreover, our method advances in diversity and cross-species generation. 



## :bulb: FAQs

- [x] None



## 💗 Acknowledgements

- Our implementation is greatly based on the [diffusers](https://github.com/huggingface/diffusers), [textual-inversion](https://github.com/rinongal/textual_inversion), [ControlNet](https://github.com/lllyasviel/ControlNet), [attention-map-diffusers](https://github.com/wooyeolbaek/attention-map-diffusers). Thanks for their wonderful works and all contributors.



## :black_nib: Citation

If you find our findings helpful in your research, please consider giving this repository a :star: and citing:

```bibtex
@misc{xuan2025rethinksparsesignalsposeguided,
      title={Rethink Sparse Signals for Pose-guided Text-to-image Generation}, 
      author={Wenjie Xuan and Jing Zhang and Juhua Liu and Bo Du and Dacheng Tao},
      year={2025},
      eprint={2506.20983},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2506.20983}, 
}
```

