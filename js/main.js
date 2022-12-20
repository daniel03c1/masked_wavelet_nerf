base_url = "https://huggingface.co/blee/Masked_Wavelet_Representation/resolve/main/images/"

function select_model(target, dir, dataset, scene) { 
    if (dir == "left"){
        var img = document.getElementById("left_img" + "_" + dataset + "_" + scene);
        var caption = document.getElementById("left_caption" + "_" + dataset + "_" + scene);
    }
    else if (dir == "right"){
        var img = document.getElementById("right_img" + "_" + dataset + "_" + scene);
        var caption = document.getElementById("right_caption" + "_" + dataset + "_" + scene);
    }
    var model = target.value;   
	var frame = document.getElementById("input" + "_" + dataset + "_" + scene).value.toString().padStart(3, '0');
	
    if (model == "TensoRF")
        img.src = base_url + "TensoRF" + "/" + frame + ".png"
    else
	    img.src = base_url + frame + "/" + model + ".png"
    
    caption.innerHTML = get_psnr(model, dataset, scene)
}


function select_frame(target, dataset, scene) {
	var left_model = document.getElementById("left_select" + "_" + dataset + "_" + scene).value;
	var right_model = document.getElementById("right_select" + "_" + dataset + "_" + scene).value;

	var left_img = document.getElementById("left_img" + "_" + dataset + "_" + scene);
	var right_img = document.getElementById("right_img" + "_" + dataset + "_" + scene);

	var frame = target.value.toString().padStart(3, '0');

    if (left_model == "TensoRF")
        left_img.src = base_url + "TensoRF" + "/" + frame + ".png"
    else
        left_img.src = base_url + frame + "/" + left_model + ".png"
        
    if (right_model == "TensoRF")
        right_img.src = base_url + "TensoRF" + "/" + frame + ".png"
    else
        right_img.src = base_url + frame + "/" + right_model + ".png"
}

function get_psnr(model, dataset, scene){
	if(model == "TensoRF"){
		if(dataset == "NSVF"){
			if(scene == "Spaceship")
				return "<b>TensoRF-VM192</b><br>PSNR: 37.66"
		}
	}
	else if(model == "1.6e-09"){
		if(dataset == "NSVF"){
			if(scene == "Spaceship")
				return "<b>Ours-Sparsity 0.9967</b><br>PSNR: 34.50"
        }
	}
	else if(model == "8e-10"){
		if(dataset == "NSVF"){
			if(scene == "Spaceship")
				return "<b>Ours-Sparsity 0.9948</b><br>PSNR: 35.60"
        }
	}
	else if(model == "4e-10"){
		if(dataset == "NSVF"){
			if(scene == "Spaceship")
				return "<b>Ours-Sparsity 0.9925</b><br>PSNR: 36.26"
        }
	}
	else if(model == "2e-10"){
		if(dataset == "NSVF"){
			if(scene == "Spaceship")
				return "<b>Ours-Sparsity 0.9886</b><br>PSNR: 36.76"
        }
	}
	else if(model == "1e-10"){
		if(dataset == "NSVF"){
			if(scene == "Spaceship")
				return "<b>Ours-Sparsity 0.9823</b><br>PSNR: 37.01"
        }
	}
	else if(model == "5e-11"){
		if(dataset == "NSVF"){
			if(scene == "Spaceship")
				return "<b>Ours-Sparsity 0.9697</b><br>PSNR: 37.35"
        }
	}
	else if(model == "2.5e-11"){
		if(dataset == "NSVF"){
			if(scene == "Spaceship")
				return "<b>Ours-Sparsity 0.9460</b><br>PSNR: 37.34"
        }
	}
	else if(model == "1.25e-11"){
		if(dataset == "NSVF"){
			if(scene == "Spaceship")
				return "<b>Ours-Sparsity 0.9165</b><br>PSNR: 37.51"
        }
	}
	else if(model == "6.25e-12"){
		if(dataset == "NSVF"){
			if(scene == "Spaceship")
				return "<b>Ours-Sparsity 0.8876</b><br>PSNR: 37.61"
        }
	}
}
