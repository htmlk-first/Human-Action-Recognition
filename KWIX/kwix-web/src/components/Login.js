import { Link, useNavigate } from "react-router-dom";
import React from "react";
import { observer } from "mobx-react";
import { auth as authApi } from "../services/api";

export const Login = observer(({ auth }) => {
	const navigate = useNavigate();
	const [formValue, setformValue] = React.useState({
		email: "",
		password: "",
	});

	const handleSubmit = async (event) => {
		event.preventDefault();
		try {
			const res = await authApi.login(formValue);
			alert(res.data.message);
			auth.login(formValue.email);
			navigate("/");
		} catch (error) {
			alert(error.response.data.message);
		}
	};

	const handleChange = (event) => {
		setformValue({
			...formValue,
			[event.target.name]: event.target.value,
		});
	};

	return (
		<div className="container">
			<div className="row align-items-center g-lg-5 py-5">
				<div className="col-md-10 mx-auto col-lg-5">
					<form
						className="p-4 p-md-5 border rounded-3 bg-light"
						onSubmit={handleSubmit}
					>
						<div className="form-floating mb-3">
							<label htmlFor="floatingInput">이메일 주소</label>
							<input
								type="email"
								className="form-control"
								id="floatingInput"
								placeholder="name@example.com"
								name="email"
								value={formValue.email}
								onChange={handleChange}
							/>
						</div>
						<div className="form-floating mb-3">
							<label htmlFor="floatingPassword">비밀번호</label>
							<input
								type="password"
								className="form-control"
								id="floatingPassword"
								placeholder="Password"
								name="password"
								value={formValue.password}
								onChange={handleChange}
							/>
						</div>
						<div className="checkbox mb-3">
							<label>
								{/* <input type="checkbox" value="remember-me">
                  {" "}
                  로그인 상태 유지
                </input> */}
							</label>
						</div>
						<button
							className="w-100 btn btn-lg btn-primary"
							type="submit"
						>
							로그인
						</button>
						<hr className="my-4" />
						<button
							className="w-100 btn btn-lg btn-primary"
							type="button"
						>
							<Link to="/signup" className="text-white">회원가입</Link>
						</button>
						<hr className="my-4" />
						<small className="text-muted">
							By clicking Sign up, you agree to the terms of use.
						</small>
					</form>
				</div>
			</div>
		</div>
	);
});
