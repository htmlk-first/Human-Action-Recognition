import React from "react";
import { Navigate } from "react-router-dom";
import { user } from "../services/api";
import { Link, useNavigate } from "react-router-dom";

export function SignUp() {
	const navigate = useNavigate();
	const [formValue, setformValue] = React.useState({
		name: "",
		birthdayDate: "",
		sex: 1,
		id: "",
		password: "",
		email: "",
		phoneNumber: "",
	});

	const handleSubmit = async (event) => {
		event.preventDefault();
		try {
			const res = await user.register(formValue);
			alert(res.data.message);
			navigate("/login");
		} catch (error) {
			alert(error.response.data.message);
		}
	};

	const handleChange = (event) => {
		event.preventDefault()
		console.log(event.target.value);
		let value = event.target.value;
		if (event.target.name === "sex") {
			value = parseInt(event.target.value, 10);
		}
		setformValue({
			...formValue,
			[event.target.name]: value,
		});
		console.log(formValue);
	};

	return (
		<div className="container py-5 h-100">
			<div className="row justify-content-center align-items-center h-100">
				<div className="col-12 col-lg-9 col-xl-7">
					<div
						className="card shadow-2-strong card-registration"
						style={{ borderRadius: 15 }}
					>
						<div className="card-body p-4 p-md-5">
							<h3 className="mb-4 pb-2 pb-md-0 mb-md-5">
								회원가입 정보 입력
							</h3>
							<form onSubmit={handleSubmit}>
								<div className="row">
									<div className="col-md-6 mb-4">
										<div className="form-outline">
											<label
												className="form-label"
												htmlFor="Name"
											>
												이름
											</label>
											<input
												type="text"
												id="Name"
												className="form-control form-control-lg"
												name="name"
												value={formValue.name}
												onChange={handleChange}
											/>
										</div>
									</div>
								</div>

								<div className="row">
									<div className="col-md-6 mb-4 d-flex align-items-center">
										<div className="form-outline datepicker w-100">
											<label
												htmlFor="birthdayDate"
												className="form-label"
											>
												생년월일(8글자)
											</label>
											<input
												type="text"
												className="form-control form-control-lg"
												id="birthdayDate"
												name="birthdayDate"
												value={formValue.birthdayDate}
												onChange={handleChange}
											/>
										</div>
									</div>
									<div className="col-md-6 mb-4">
										<h6 className="mb-2 pb-1">성별: </h6>

										<div className="form-check form-check-inline">
											<input
												className="form-check-input"
												type="radio"
												name="sex"
												id="maleGender"
												checked={formValue.sex === 1}
												value={1}
												onChange={handleChange}
											/>
											<label
												className="form-check-label"
												htmlFor="maleGender"
											>
												남자
											</label>
										</div>

										<div className="form-check form-check-inline">
											<input
												className="form-check-input"
												type="radio"
												name="sex"
												id="femaleGender"
												checked={formValue.sex === 0}
												value={0}
												onChange={handleChange}
											/>
											<label
												className="form-check-label"
												htmlFor="femaleGender"
											>
												여자
											</label>
										</div>
									</div>
								</div>

								<div className="row">
									<div className="col-md-6 mb-4">
										<div className="form-outline">
											<label
												className="form-label"
												htmlFor="id_"
											>
												아이디
											</label>
											<input
												type="text"
												id="id_"
												className="form-control form-control-lg"
												name="id"
												value={formValue.id}
												onChange={handleChange}
											/>
										</div>
									</div>
								</div>

								<div className="row">
									<div className="col-md-6 mb-4">
										<div className="form-outline">
											<label
												className="form-label"
												htmlFor="password"
											>
												비밀번호
											</label>
											<input
												type="text"
												id="password"
												className="form-control form-control-lg"
												name="password"
												value={formValue.password}
												onChange={handleChange}
											/>
										</div>
									</div>
								</div>

								<div className="row">
									<div className="col-md-6 mb-4 pb-2">
										<div className="form-outline">
											<label
												className="form-label"
												htmlFor="emailAddress"
											>
												Email
											</label>
											<input
												type="email"
												id="emailAddress"
												className="form-control form-control-lg"
												name="email"
												value={formValue.email}
												onChange={handleChange}
											/>
										</div>
									</div>
									<div className="col-md-6 mb-4 pb-2">
										<div className="form-outline">
											<label
												className="form-label"
												htmlFor="phoneNumber"
											>
												Phone Number
											</label>
											<input
												type="tel"
												id="phoneNumber"
												className="form-control form-control-lg"
												name="phoneNumber"
												value={formValue.phoneNumber}
												onChange={handleChange}
											/>
										</div>
									</div>
								</div>

								<div className="mt-4 pt-2">
									<button
										className="w-50 btn btn-lg btn-primary"
										type="submit"
									>
										회원가입
									</button>
								</div>
							</form>
						</div>
					</div>
				</div>
			</div>
		</div>
	);
}
