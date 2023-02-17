import React from "react";
import { useNavigate } from "react-router-dom";
import { recommend } from "../services/api";

const StartBtn = () => {
	const navigate = useNavigate();

	const click = async (event) => {
		try {
			const email = localStorage.getItem("kwixUser");
			await recommend(email);
			navigate("/recommend"); 
		} catch (error) {
			alert(error.response.data.message);
			navigate("/input");
		}
	};

	return (
		<button onClick={click} className="btn-solid-lg page-scroll">
			Start
		</button>
	);
};

export function Header() {
	return (
		<header id="header" className="header">
			<div className="container">
				<div className="row">
					<div className="col-lg-6">
						<div className="text-container">
							<h1 className="h1-large">
								Human activity recognition and recommendation system
							</h1>
							{StartBtn()}
						</div>
					</div>
				</div>
			</div>
		</header>
	);
}
