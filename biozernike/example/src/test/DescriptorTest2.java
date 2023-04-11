
import org.apache.commons.lang.ArrayUtils;
import org.biojava.nbio.structure.*;
import org.biojava.nbio.structure.align.util.AtomCache;
import org.biojava.nbio.structure.io.FileParsingParameters;
import org.biojava.nbio.structure.quaternary.BioAssemblyTools;
import org.biojava.nbio.structure.quaternary.BiologicalAssemblyBuilder;
import org.biojava.nbio.structure.quaternary.BiologicalAssemblyTransformation;
import org.junit.BeforeClass;
import org.junit.Test;
import org.rcsb.biozernike.complex.Complex;
import org.rcsb.biozernike.descriptor.Descriptor;
import org.rcsb.biozernike.descriptor.DescriptorConfig;
import org.rcsb.biozernike.descriptor.DescriptorMode;
import org.rcsb.biozernike.volume.MapFileType;
import org.rcsb.biozernike.volume.Volume;
import org.rcsb.biozernike.volume.VolumeIO;
import org.rcsb.biozernike.zernike.ZernikeMoments;

import javax.vecmath.Matrix4d;
import javax.vecmath.Point3d;
import java.io.InputStream;
import java.io.PrintWriter;
import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.EnumSet;
import java.util.List;
import java.util.function.Function;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Collections;

final class utils {

	public static boolean exist(String path) {
		File file = new File(path);

		if (file.exists()) {
			return true;
		} else {
			return false;
		}
	}

	public static String[] read(String filePath) throws IOException {
		List<String> pdbList = new ArrayList<String>();

		BufferedReader reader = new BufferedReader(new FileReader(filePath));
		String line;
		while ((line = reader.readLine()) != null) {
			String pdbId = line.trim().toUpperCase();
			if (pdbId.length() >= 4) {
				pdbList.add(pdbId);
			}
		}
		reader.close();

		return pdbList.toArray(new String[0]);
	}
}

public class DescriptorTest2 {

	@BeforeClass
	public static void setupBioJava() {
		FileParsingParameters params = new FileParsingParameters();
		params.setParseBioAssembly(true);
		AtomCache cache = new AtomCache();
		cache.setFileParsingParams(params);
		StructureIO.setAtomCache(cache);
	}

	// @Test
	// Run a program that inputs pdb file
	// convert protein into structure
	// and then pick out the atoms
	// and then convert them into points
	// and get the geometry descriptor and moment descriptor

	// the pdb file cache is in PDB_DIR, but I am too lazy to collect them... So let
	// the program download them

	// note that some entries have been obsolote and cannot be downloaded, you need
	// to download them with their newer entry
	public void convertPDB() throws Exception {
		String[] pdbList = utils.read("src/test/pdbid.txt");
		boolean override = false;
		String workingdir = "z:/pdb/";
		Collections.shuffle(Arrays.asList(pdbList));
		// open a file to record the pdbids that cannot be fetched
		PrintWriter writer3 = new PrintWriter(new FileWriter(workingdir + "error.txt", true));
		for (int j = 0; j < pdbList.length; j++) {
			String pdbentry = pdbList[j];
			// do not overwrite existing files
			if (!override && utils.exist(workingdir + pdbentry + ".moment")
					&& utils.exist(workingdir + pdbentry + ".geo")) {
				continue;
			}
			// check if the existing file *.moment is old enough
			File file = new File(workingdir + pdbentry + ".moment");
			if (file.exists()) {
				long lastModified = file.lastModified();
				long currentTime = System.currentTimeMillis();
				// if the file is within 1 hour, continue
				if (currentTime - lastModified < 3600000) {
					continue;
				}
			}
			// first write an empty file
			PrintWriter writer0 = new PrintWriter(workingdir + pdbentry + ".moment",
					"UTF-8");
			// write a string
			writer0.println("empty");
			writer0.close();
			BiologicalAssemblyBuilder builder = new BiologicalAssemblyBuilder();
			// if the structure is not available, write it to error, and delete the empty
			// file
			Structure structure;
			try {
				structure = StructureIO.getStructure(pdbentry);
			} catch (Exception e) {
				writer3.println(pdbentry);
				File file2 = new File(workingdir + pdbentry + ".moment");
				file2.delete();
				continue;
			}
			List<BiologicalAssemblyTransformation> transformations = structure.getPDBHeader().getBioAssemblies().get(1)
					.getTransforms();

			Structure bioUnitStructure = builder
					.rebuildQuaternaryStructure(BioAssemblyTools.getReducedStructure(structure),
							transformations, true,
							false);

			Atom[] reprAtoms = StructureTools.getRepresentativeAtomArray(bioUnitStructure);
			Point3d[] reprPoints = Calc.atomsToPoints(reprAtoms);
			String[] resNames = new String[reprAtoms.length];
			for (int i = 0; i < reprAtoms.length; i++) {
				final Atom a = reprAtoms[i];
				resNames[i] = new Function<Atom, String>() {
					@Override
					public String apply(Atom a) {
						return a.getGroup().getPDBName();
					}
				}.apply(a);
			}
			EnumSet<DescriptorMode> mode = EnumSet.allOf(DescriptorMode.class);
			DescriptorConfig config = new DescriptorConfig(
					DescriptorTest2.class.getResourceAsStream("/descriptor.properties"), mode);

			Descriptor ssd = new Descriptor(reprPoints, resNames, config);
			double[] MomentDescriptors = ssd.getMomentDescriptor();
			double[] GeometryDescriptors = ssd.getGeometryDescriptor();
			// save both to a file
			PrintWriter writer = new PrintWriter(workingdir + pdbentry + ".moment",
					"UTF-8");
			for (int i = 0; i < GeometryDescriptors.length; i++) {
				writer.println(GeometryDescriptors[i]);
			}
			writer.close();

			PrintWriter writer2 = new PrintWriter(workingdir + pdbentry + ".geo",
					"UTF-8");
			for (int i = 0; i < MomentDescriptors.length; i++) {
				writer2.println(MomentDescriptors[i]);
			}
			writer2.close();
		}
		writer3.close();
	}

	@Test
	public void convertObsolotePDB() throws Exception {
		String[] obsolote = utils.read("src/test/obsolote.txt");
		boolean override = false;
		String workingdir = "z:/pdb/";
		// obsolote string is in the format of "old=new"
		// try download old while save to new
		// the procedure is the same as convertPDB
		for (int i = 0; i < obsolote.length; i++) {
			String[] temp = obsolote[i].split("=");
			String old = temp[0];
			String _new = temp[1];
			if (!override && utils.exist(workingdir + old + ".moment")
					&& utils.exist(workingdir + old + ".geo")) {
				continue;
			}
			// check if the existing file *.moment is old enough
			File file = new File(workingdir + old + ".moment");
			if (file.exists()) {
				long lastModified = file.lastModified();
				long currentTime = System.currentTimeMillis();
				// if the file is within 1 hour, continue
				if (currentTime - lastModified < 3600000) {
					continue;
				}
			}
			// first write an empty file
			PrintWriter writer0 = new PrintWriter(workingdir + old + ".moment",
					"UTF-8");
			// write a string
			writer0.println("empty");
			writer0.close();
			BiologicalAssemblyBuilder builder = new BiologicalAssemblyBuilder();
			// if the structure is not available, write it to error, and delete the empty
			// file
			Structure structure;
			try {
				structure = StructureIO.getStructure(_new);
			} catch (Exception e) {
				// print to screen
				System.out.println("ERROR!!!!!!!!!!!!!!!!!!!!!!!!!!!!"+old);
				System.out.println(e.getMessage());
				File file2 = new File(workingdir + old + ".moment");
				file2.delete();
				continue;
			}
			List<BiologicalAssemblyTransformation> transformations = structure.getPDBHeader().getBioAssemblies().get(1)
					.getTransforms();

			Structure bioUnitStructure = builder
					.rebuildQuaternaryStructure(BioAssemblyTools.getReducedStructure(structure),
							transformations, true,
							false);

			Atom[] reprAtoms = StructureTools.getRepresentativeAtomArray(bioUnitStructure);
			Point3d[] reprPoints = Calc.atomsToPoints(reprAtoms);
			String[] resNames = new String[reprAtoms.length];
			for (int k = 0; k < reprAtoms.length; k++) {
				final Atom a = reprAtoms[k];
				resNames[k] = new Function<Atom, String>() {
					@Override
					public String apply(Atom a) {
						return a.getGroup().getPDBName();
					}
				}.apply(a);
			}
			String pdbentry = old;
			EnumSet<DescriptorMode> mode = EnumSet.allOf(DescriptorMode.class);
			DescriptorConfig config = new DescriptorConfig(
					DescriptorTest2.class.getResourceAsStream("/descriptor.properties"), mode);

			Descriptor ssd = new Descriptor(reprPoints, resNames, config);
			double[] MomentDescriptors = ssd.getMomentDescriptor();
			double[] GeometryDescriptors = ssd.getGeometryDescriptor();
			// save both to a file
			PrintWriter writer = new PrintWriter(workingdir + pdbentry + ".moment",
					"UTF-8");
			for (int j = 0; j < GeometryDescriptors.length; j++) {
				writer.println(GeometryDescriptors[j]);
			}
			writer.close();

			PrintWriter writer2 = new PrintWriter(workingdir + pdbentry + ".geo",
					"UTF-8");
			for (int l = 0; l < MomentDescriptors.length; l++) {
				writer2.println(MomentDescriptors[l]);
			}
			writer2.close();
		}
	}
}
