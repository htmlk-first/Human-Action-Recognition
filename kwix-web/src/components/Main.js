import { Routes, Route } from "react-router-dom";

import "./styles.css";

import { Home } from "./Home";
import { Input } from "./Input";
import { Login } from "./Login";
import { Header } from "./Header";
import { Footer } from "./Footer";
import { Nav } from "./Nav";
import { SignUp } from "./SignUp";
import { Recommend } from "./Recommend";
import { AuthStore } from "../stores/authStore";
import { User } from "./User";

const auth = new AuthStore();

export function Main() {
	return (
		<div data-spy="scroll" data-target=".fixed-top">
			<Nav auth={auth}></Nav>
			<Header></Header>

			<div id="services" className="basic-2">
				<Routes>
					<Route path="/" element={<Home />} />
					<Route path="/input" element={<Input />} />
					<Route path="/login" element={<Login auth={auth} />} />
					<Route path="/signup" element={<SignUp />} />
					<Route path="/recommend" element={<Recommend />} />
					<Route path="/user" element={<User auth={auth} />} />
				</Routes>
			</div>

			<Footer></Footer>
		</div>
	);
}
