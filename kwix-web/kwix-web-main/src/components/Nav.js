import { Link, useNavigate } from "react-router-dom";
import React from "react";
import { observer } from "mobx-react";
import { auth as authApi } from "../services/api";

export const Nav = observer(({ auth }) => {
	const navigate = useNavigate();

	const handleSubmit = async (event) => {
		event.preventDefault();
		try {
			const res = await authApi.logout();
			alert(res.data.message);
			navigate("/");
			auth.logout();
		} catch (error) {
			alert(error.response.data.message);
		}
	};
	let button;
	if (auth.isAuth) {
		button = (
			<Link
				to="/"
				className="nav-link page-scroll"
				onClick={handleSubmit}
			>
				logout
			</Link>
		); 
	} else {
		button = (
			<Link to="/login" className="nav-link page-scroll">
				login
			</Link>
		);
	}
 
	return (
		<nav className="navbar navbar-expand-lg fixed-top navbar-dark">
			<div className="container">
				<Link to="/" className="navbar-brand logo-text page-scroll">
					KWIX
				</Link>
				<button
					className="navbar-toggler p-0 border-0"
					type="button"
					data-toggle="offcanvas"
				>
					<span className="navbar-toggler-icon"></span>
				</button>

				<div
					className="navbar-collapse offcanvas-collapse"
					id="navbarsExampleDefault"
				>
					<ul className="navbar-nav ml-auto">
						<li className="nav-item">
							<a className="nav-link page-scroll" href="#header">
								Home <span className="sr-only">(current)</span>
							</a>
						</li>
						{/* <li className="nav-item">
							<a
								className="nav-link page-scroll"
								href="#services"
							>
								Services
							</a>
						</li> */}

						<li className="nav-item">{button}</li>
						<li className="nav-item">{UserLink(auth.isAuth)}</li>
					</ul>
					<span className="nav-item social-icons">
						<span className="fa-stack">
							<a href="#your-link">
								<i className="fas fa-circle fa-stack-2x"></i>
								<i className="fab fa-facebook-f fa-stack-1x"></i>
							</a>
						</span>
						<span className="fa-stack">
							<a href="#your-link">
								<i className="fas fa-circle fa-stack-2x"></i>
								<i className="fab fa-twitter fa-stack-1x"></i>
							</a>
						</span>
					</span>
				</div>
			</div>
		</nav>
	);
});

/**
 *
 * @param {boolean} isAuth
 * @returns
 */
const UserLink = (isAuth) => {
	if (isAuth) {
		return (
			<div>
				<Link to="/user" className="nav-link page-scroll">
					User
				</Link>
			</div>
		);
	}

	return;
};
